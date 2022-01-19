# This script hosts general utility functions to perform route shape matching.
# The code uses lower cased column names.

# Packages
import logging
import geopandas as gpd
from shapely.geometry import box, LineString
import osmnx as ox
from datetime import date
import os
import pandas as pd
import networkx as nx
import numpy as np

# Parameters
CRS = "EPSG:4326"
CRS_METER = "EPSG:3857"

# Configure logger
logging.basicConfig(level=logging.CRITICAL,
                    format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('WMATA')
logger.setLevel(logging.INFO)
logger.info('WMATA Route Shape Matching.')


# Functions
def get_today():
    """
    Get current date.
    :return: Current date in YYYYMMDD format.
    """
    return date.today().strftime("%Y%m%d")


def read_all_gis_route_lines(gis_lines_geojson,
                             crs):
    """
    Get all GIS route line geometries in the data.

    :param gis_lines_geojson: Directory of GIS bus route geojson file including file name and extension.
    :param crs: Coordinate Reference System in "EPSG:XXXX" format.
    :return: A GeoDataFrame with line geometries for all available route patterns in specified CRS.
    """

    # Import WMATA GIS bus route line geometries
    gis_routes = gpd.read_file(gis_lines_geojson)
    gis_routes = gis_routes.to_crs(crs)
    gis_routes.columns = [c.lower() if c != 'geometry' else c for c in gis_routes.columns]

    # Create "shape_id" column
    gis_routes['pattern_id'] = gis_routes['gis_routec'].str.split('_').str[-1]
    gis_routes['shape_id'] = gis_routes['route'] + gis_routes['pattern_id']
    gis_routes = gis_routes.sort_values(by='shape_id')

    # Create start and end date in YYYYMM format for schedule based selection
    gis_routes['start_date'] = gis_routes['str_date'].astype(str).str[:10].str.replace('-', '').astype(int)
    gis_routes['end_date'] = gis_routes['end_date'].astype(str).str[:10].str.replace('-', '').astype(int)
    logger.info(f'Route line geometries returned for {gis_routes.shape_id.nunique():,} unique route patterns.')

    return gis_routes.sort_values(by=['start_date'])


def get_single_gis_route_line(gis_routes,
                              route_pattern_id,
                              query_month):
    """
    Get single GIS line geometry.

    :param gis_routes: A GeoDataFrame with line geometries for all available route patterns.
    :param route_pattern_id: Candidate route pattern ID in RRDD format (RR = Route Number, DD = Direction).
    :param query_month: Query date in YYYYMMDD format - must be in integer.
    :return: Relevant rows from the GIS bus route geojson with line geometries in specified CRS.
    """
    single_route = gis_routes[(gis_routes['shape_id'] == route_pattern_id) &
                              (gis_routes['start_date'] <= query_month) &
                              (gis_routes['end_date'] > query_month)].sort_values(by=['start_date'])
    logger.info(f'Route line geometries returned for {route_pattern_id}.')
    return single_route


def read_single_gis_route_line(gis_lines_geojson,
                               crs,
                               route_pattern_id,
                               query_month):
    """
    Read single GIS line geometry.

    :param gis_lines_geojson: Directory of GIS bus route geojson file including file name and extension.
    :param crs: Coordinate Reference System in "EPSG:XXXX" format.
    :param route_pattern_id: Candidate route pattern ID in RRDD format (RR = Route Number, DD = Direction).
    :param query_month: Query date in YYYYMMDD format - must be in integer.
    :return: Relevant rows from the GIS bus route geojson with line geometries in specified CRS.
    """
    # Import WMATA GIS bus route line geometries
    gis_routes = read_all_gis_route_lines(gis_lines_geojson, crs)
    single_route = get_single_gis_route_line(gis_routes,
                                             route_pattern_id,
                                             query_month)
    return single_route


def read_all_routes_stop_sequences(stop_seq_csv, crs):
    """
    Get stop sequences for all route patterns.
    :param stop_seq_csv: Directory of stop sequence CSV file including file name and extension.
    :param crs: Coordinate Reference System in "EPSG:XXXX" format.
    :return: Stop sequence GeoDataFrame.
    """
    stop_seq = pd.read_csv(stop_seq_csv)
    stop_seq.columns = [c.lower() for c in stop_seq.columns]
    stop_seq['shape_id'] = stop_seq['pattern_id']
    logger.info(f'Stop sequences returned for {stop_seq.shape_id.nunique():,} route patterns.')
    return create_point_gdf(stop_seq, lon_col='longitude', lat_col='latitude', crs=crs)


def get_single_route_stop_sequences(stop_seq, route_pattern_id):
    """
    Get stop sequences for candidate route pattern.
    :param stop_seq: Stop sequence GeoDataFrame.
    :param route_pattern_id: Candidate route pattern ID in RRDD format (RR = Route Number, DD = Direction).
    :return: Stop sequence GeoDataFrame for candidate route pattern.
    """
    gdf = stop_seq[stop_seq.shape_id == route_pattern_id].sort_values(by=['stop_sequence'])
    logger.info(f'Stop sequences returned for {route_pattern_id} with {len(gdf):,} stops.')
    return gdf


def get_unified_polygon(gdf,
                        crs,
                        buffer_threshold):
    """
    Union all geometries together and buffer the dissolved boundaries.
    Note: The buffer radius must be in the units of the coordinate reference system.

    :param gdf: A GeoDataFrame.
    :param crs: Coordinate Reference System in "EPSG:XXXX" format.
    :param buffer_threshold: Buffer distance in the unit of CRS.
    :return: A unified polygon geometry created from the given GeoDataFrame.
    """

    gdf = gdf.to_crs(crs)
    gdf.loc[:, 'geometry'] = (gdf.loc[:, 'geometry']
                              .apply(lambda x: box(x.bounds[0], x.bounds[1], x.bounds[2], x.bounds[3])))
    return gdf.unary_union.buffer(buffer_threshold)


def download_osm_for_transit(polygon_geometry, log_console=True):
    """
    Download the OSM street network for given polygon
    and creates a Networkx graph object.

    :param polygon_geometry: A Shapely polygon object that defines the area for map extraction.
    :param log_console: Boolean indicator to specify if OSM logs should be printed on the console.
    :return: A Networkx graph object with street network.
    """

    custom_filter = (f'["highway"]["area"!~"yes"]["highway"!~"cycleway|footway|path|pedestrian|steps|track|'
                     f'corridor|elevator|escalator|proposed|construction|bridleway|abandoned|platform|raceway"]'
                     f'["motor_vehicle"!~"no"]["motorcar"!~"no"]'
                     f'["service"!~"emergency_access"]')

    ox.config(use_cache=True, log_console=log_console)
    return ox.graph_from_polygon(polygon_geometry,
                                 network_type='drive_service',
                                 custom_filter=custom_filter)


def export_osm_graph_shapefiles(graph_obj,
                                output_path,
                                export_nodes=True,
                                export_edges=True):
    """
    Export OSM graph and nodes/edges GeoDataFrames if specified.
    :param graph_obj: A graph object of OSM street network.
    :param output_path: Output directory.
    :param export_nodes: Boolean indicator to specify if nodes GeoDataFrame should be exported.
    :param export_edges: Boolean indicator to specify if edges GeoDataFrame should be exported.
    :return: None
    """
    ox.save_graphml(graph_obj, os.path.join(output_path, f'{get_today()}_osm_graph.graphml'))

    # Nodes and edges from graph
    nodes, edges = get_graph_nodes_edges_sindex(graph_obj, spatial_index=False)
    if export_nodes:
        nodes.to_file(os.path.join(output_path, f'{get_today()}_osm_graph_nodes.geojson'))
    if export_edges:
        # Clean before exporting
        clean_edge_fields_for_export(edges).to_file(os.path.join(output_path, f'{get_today()}_osm_graph_edges.geojson'))


def get_spatial_index(gdf):
    """
    Get spatial indexes of spatial objects from a GeoDataFrame.
    :param gdf: A GeoDataFrame.
    :return: Spatial indexes of spatial objects.
    """
    return gdf.sindex


def get_graph_nodes_edges_sindex(graph_obj, spatial_index=True):
    """
    Create nodes and edges from graph with spatial indexes if specified.
    :param graph_obj: A graph object of street network.
    :param spatial_index: Boolean indicator to specify if spatial indexes should be returned as well.
    :return: Nodes and edges GeoDataFrame with nodes and edges spatial indexes if specified.
    """
    nodes, edges = ox.graph_to_gdfs(graph_obj, edges=True)
    logger.info(f'OSM graph converted to nodes ({len(nodes):,}) and edges ({len(edges):,}) GeoDataFrames.')

    if spatial_index:
        node_sindex = get_spatial_index(nodes)
        edge_sindex = get_spatial_index(edges)
        logger.info(f'Spatial indexes returned for {len(nodes):,} nodes and {len(edges):,} edges.')
        return nodes.reset_index(), edges.reset_index(), node_sindex, edge_sindex
    else:
        return nodes.reset_index(), edges.reset_index()


def convert_list_to_string(df, col):
    """
    Convert list-type cell value to string-type cell value.

    :param df: A DataFrame or GeoDataFrame.
    :param col: A list of columns with list-type cell values.
    :return: A DataFrame or GeoDataFrame with list-type cell values converted to string-type cell values.
    """
    return df.loc[:, col].apply(lambda x: str(x) if type(x) != list else ", ".join([str(item) for item in x]))


def clean_edge_fields_for_export(edge_df):
    """
    Convert list-type cell values to string-type cell values in edge fields for export.

    :param edge_df: An edge DataFrame or GeoDataFrame.
    :return: Clean DataFrame or GeoDataFrame for export.
    """
    col_list = ['osmid', 'name', 'highway', 'length', 'oneway']
    for col in col_list:
        edge_df.loc[:, f'{col}'] = convert_list_to_string(edge_df, col)
    return edge_df[['u', 'v'] + col_list + ['geometry']].copy()


def create_point_gdf(df, lon_col, lat_col, crs):
    """
    Create a point GeoDataFrame from longitude/latitude coordinates.
    :param df: DataFrame with longitude/latitude columns.
    :param lon_col: Name of longitude column.
    :param lat_col: Name of latitude column.
    :param crs: Coordinate Reference System in "EPSG:XXXX" format.
    :return: Point GeoDataFrame.
    """
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs=crs)


def read_gtfs_shapes(gtfs_shapes_text, crs):
    """
    Read shape points from GTFS "shapes.txt" file.
    :param gtfs_shapes_text: File path for GTFS "shapes.txt" (including file name and extension).
    :param crs: Coordinate Reference System in "EPSG:XXXX" format.
    :return: Shape points GeoDataFrame for all route patterns.
    """
    shapes = pd.read_csv(gtfs_shapes_text)
    shapes.columns = [c.lower() for c in shapes.columns]
    shapes['shape_id'] = shapes['shape_id'].str.replace(':', '')
    shapes = create_point_gdf(shapes, 'shape_pt_lon', 'shape_pt_lat', crs)
    logger.info(f'Route shape geometries returned for {shapes.shape_id.nunique():,} '
                f'route patterns with {len(shapes):,} points.')
    return shapes


def get_single_route_gtfs_shapes(shapes, route_pattern_id):
    """
    Get GTFS shape points for a single route pattern.
    :param shapes: Shape points GeoDataFrame for all route patterns.
    :param route_pattern_id: Candidate route pattern ID in RRDD format (RR = Route Number, DD = Direction).
    :return: Shape points GeoDataFrame for candidate route pattern.
    """
    shapes = shapes[shapes.shape_id == route_pattern_id]
    logger.info(f'Route shape geometries returned for {route_pattern_id} with {len(shapes):,} points.')
    return shapes


def get_line_geom_from_shapes(shapes_gdf, crs=CRS):
    """
    Create line geometry from shape points.
    :param shapes_gdf: Shape points GeoDataFrame for candidate route pattern.
    :param crs: Coordinate Reference System in "EPSG:XXXX" format.
    :return: GeoDataFrame with line geometry for candidate route pattern.
    """
    shape_line_geometry = (shapes_gdf.sort_values(by=['shape_id', 'shape_pt_sequence'])
                           .groupby(['shape_id'])['geometry']
                           .apply(lambda x: LineString(x.tolist())))
    logger.info(f'Route line geometries returned for {shapes_gdf.shape_id.unique()[0]} '
                f'using {len(shapes_gdf):,} points.')
    return gpd.GeoDataFrame(shape_line_geometry,
                            geometry='geometry',
                            crs=crs).reset_index()


def create_buffer(gdf, buffer_in_meter, output_crs, sort_columns=None):
    """
    Create a buffer using the geometries in a GeoDataFrame.
    :param gdf: GeoDataFrame with line geometries.
    :param buffer_in_meter: Buffer distance in meters.
    :param output_crs: Output Coordinate Reference System in "EPSG:XXXX" format.
    :param sort_columns: List of columns used to sort the data.
    :return:
    """
    # Sort records if column names are given
    if sort_columns is not None:
        gdf = gdf.copy().sort_values(by=sort_columns)
    else:
        gdf = gdf.copy()

    # Convert CRS to use meter
    gdf = gdf.to_crs('EPSG:3857')
    gdf['geometry'] = gdf.geometry.buffer(buffer_in_meter)
    return gdf.to_crs(output_crs)


def create_union_of_buffer_geom(buff1, buff2, only_geom=False):
    """
    Create a union of buffers.
    :param buff1: First buffer GeoDataFrame.
    :param buff2: Second buffer GeoDataFrame.
    :param only_geom: True if only geometry should be returned.
    :return: Unified buffer geometry.
    """
    buff = gpd.overlay(buff1, buff2, how='union').dissolve()
    logger.info(f'Route buffer geometry created.')
    if only_geom:
        return buff.geometry[0]
    else:
        return buff.geometry[0], buff


def get_node_list_using_edges_within_buffer(edges, buffer_geom):
    """
    Get nodes inside the buffer.
    :param edges: Edges GeoDataFrame for the entire network.
    :param buffer_geom: Unified buffer geometry.
    :return: List of nodes that are located inside the route buffer.
    """
    # Get spatial indexes of edges
    edge_sindex = edges.sindex

    # Candidate edges
    edge_candidate_idx = list(edge_sindex.intersection(buffer_geom.bounds))
    candidate_edges = edges.iloc[edge_candidate_idx]
    intersecting_edges = candidate_edges[candidate_edges.intersects(buffer_geom)]
    u_list = intersecting_edges.loc[:, 'u'].unique().tolist()
    v_list = intersecting_edges.loc[:, 'v'].unique().tolist()
    node_list = list(set(u_list + v_list))
    logger.info(f'{len(intersecting_edges):,} edges (of {len(edges):,} total edges) found inside the buffer.')
    logger.info(f'{len(node_list):,} nodes returned.')
    return node_list


def get_subgraph_using_node_list(graph_obj, node_list):
    """
    Create a subgraph of a graph using nodes.
    :param graph_obj: Graph object.
    :param node_list: List of nodes to use for extracting the subgraph.
    :return: Subgraph of graph.
    """
    return graph_obj.subgraph(node_list).copy()


def add_nearest_edges_at_terminals(shapes, graph_obj_sub, edge_list):
    """
    Find the final list of edges including the nearest edges at the
    start and end point of the route. This is done to make sure we always capture
    the OSM edges near the start and end point of the route.
    :param shapes: Shape points GeoDataFrame for candidate route pattern.
    :param graph_obj_sub: Subgraph as graph object.
    :param edge_list: List of edges in (u, v, key) format.
    :return: Final list of edges to create the largest connected graph.
    """
    # Get first and last edges
    first_pt_geom = shapes.reset_index(drop=True).iloc[0].geometry
    last_pt_geom = shapes.reset_index(drop=True).iloc[-1].geometry

    # Using the first and last shape geometry find the nearest edges.
    first_pt_edge = ox.get_nearest_edge(graph_obj_sub,
                                        (first_pt_geom.y, first_pt_geom.x),
                                        return_geom=False,
                                        return_dist=False)

    last_pt_edge = ox.get_nearest_edge(graph_obj_sub,
                                       (last_pt_geom.y, last_pt_geom.x),
                                       return_geom=False,
                                       return_dist=False)

    first_pt_edge_opposite = (first_pt_edge[1], first_pt_edge[0], first_pt_edge[2])
    last_pt_edge_opposite = (last_pt_edge[1], last_pt_edge[0], last_pt_edge[2])
    first_last_edge_list = edge_list + [first_pt_edge, first_pt_edge_opposite,
                                        last_pt_edge, last_pt_edge_opposite]

    return first_last_edge_list


def get_hardclipped_edges(graph_obj, shapes, edges, buffer_geom, return_nodes=False):
    """
    Get the final edges within the buffer for route shape matching.
    Corresponding steps are:
        - Select nodes inside the buffer.
        - Create a subgraph using the select nodes.
        - Get edges of the subgraph.
        - Clip the edges by route buffer geometry.
        - Select candidate edges comparing the clipped length to actual length.
        - Add the nearest edges to start and end point of the route.
        - Create the largest connected graph with the final candidate edge list.
    :param graph_obj: Entire OSM network as a graph object.
    :param shapes: Shape points GeoDataFrame for candidate route pattern.
    :param edges: Edges GeoDataFrame extracted from the OSM graph object.
    :param buffer_geom: Unified buffer geometry.
    :param return_nodes: If True, final nodes GeoDataFrame should be returned as the third return object.
    :return: Final subgraph and edges GeoDataFrame for route shape matching.
    """

    # Create subgraph using nodes inside the buffer
    node_list = get_node_list_using_edges_within_buffer(edges, buffer_geom)
    g = get_subgraph_using_node_list(graph_obj, node_list)
    ge = ox.graph_to_gdfs(g, edges=True, nodes=False).reset_index()

    # Clip the edges based on route buffer
    ge_clipped = gpd.clip(ge, buffer_geom).copy()

    # Subgraph in meter
    ge_m = ge.to_crs(CRS_METER)
    ge_m['orig_length'] = ge_m.geometry.length

    # Convert crs to use meter
    ge_clipped_m = ge_clipped.to_crs(CRS_METER)

    # Merge original length before clipping
    ge_clipped_m = (ge_clipped_m
                    .merge(ge_m[['u', 'v', 'orig_length']],
                           on=['u', 'v'],
                           how='left'))
    # Add new columns to compare lengths
    ge_clipped_m.loc[:, 'new_length'] = ge_clipped_m.geometry.length
    ge_clipped_m.loc[:, 'compare_length_value'] = (ge_clipped_m['orig_length'].astype(float) * 0.01)
    ge_clipped_m.loc[:, 'new_length'].fillna(1, inplace=True)

    condition = ((abs(ge_clipped_m.new_length - ge_clipped_m['orig_length'].astype(float)))
                 < ge_clipped_m.compare_length_value)
    ge_clipped_m.loc[:, 'org_vs_new_length'] = np.where(condition, 1, 1500)

    # Hard-clip the edges
    ge_hardclipped = ge_clipped_m[ge_clipped_m['org_vs_new_length'] == 1]
    ge_hardclipped['edge_key'] = tuple(zip(ge_hardclipped['u'],
                                           ge_hardclipped['v'],
                                           ge_hardclipped['key']))

    ge_hardclipped_list = ge_hardclipped['edge_key'].tolist()

    # Add the nearest edges near the start and end point of the route
    final_edge_list = add_nearest_edges_at_terminals(shapes, g, ge_hardclipped_list)

    # Get the largest connected graph and add bearings
    g_hardclipped = ox.utils_graph.get_largest_component(nx.edge_subgraph(graph_obj, final_edge_list))
    g_hardclipped = ox.add_edge_bearings(g_hardclipped)

    # Export hardclipped nodes and edges
    g_hardclipped_nodes, g_hardclipped_edges = get_graph_nodes_edges_sindex(g_hardclipped, spatial_index=False)
    logger.info(f'{len(g_hardclipped_edges):,} edges returned after final clipping.')
    if return_nodes:
        return g_hardclipped, g_hardclipped_edges, g_hardclipped_nodes
    else:
        return g_hardclipped, g_hardclipped_edges


def get_bearing_difference(b1, b2):
    """
    Calculate absolute difference between two bearings.
    :param b1: First bearing value.
    :param b2: Second bearing value.
    :return: Absolute differences between bearings.
    """
    b = abs(b1 - b2) % 360
    if b > 180:
        return 360 - b
    else:
        return b


def construct_path(od_shapes, edges_hardclipped):
    """
    Construct route path using candidate edges. This helps
    remove cul-de-sac or wrongs edges obtained from nearest edge search results.
    :param od_shapes: DataFrame with edge codes (u, v) returned after finding shape-shape shortest paths.
    :param edges_hardclipped: Edges (hard-clipped) GeoDataFrame for route shape matching.
    :return: GeoDataFrame with sequential edges geometries.
    """
    path = list(zip(od_shapes.u, od_shapes.v))

    searchables = path[1:].copy()
    segment = [path[0]]
    i = path[0]

    def find_next(m, s):
        for n in s:
            if m[1] == n[0]:
                return n

    while len(searchables) > 0:
        j = find_next(i, searchables)
        if j is not None:
            segment.append(j)
            i = j
            searchables = searchables[1:]
        else:
            segment = segment[:-1]
            i = segment[-1]

    # Create DataFrame
    route_path = pd.DataFrame({'u': [x[0] for x in segment], 'v': [x[1] for x in segment]})
    route_path['edge_code'] = route_path.u.astype(str) + '-' + route_path.v.astype(str)
    route_path['edge_order'] = [*range(1, len(route_path) + 1)]
    route_path = (edges_hardclipped[['u', 'v', 'geometry']].to_crs(CRS_METER)
                  .merge(route_path, on=['u', 'v'], how='right')
                  .drop_duplicates('geometry'))
    route_path['edge_len_ft'] = route_path.geometry.length * 3.28084
    return route_path


def get_route_path(shapes, graph_obj_h, edges_hardclipped):
    """
    Get route path using the shortest path between two shape points.
    Corresponding steps are:
        - Using the subgraph, find the nearest edge to the start and end point of each shape-shape pair.
        - Correct orientation of returned nearest edges using bearing values.
        - Get the shortest path for each shape-shape in sequential order and keep unique segments.
        - Get the nodes in correct order using forward search method.
        - Construct the route path.
    :param shapes: Shape points GeoDataFrame for candidate route pattern.
    :param graph_obj_h: Subgraph (hard-clipped).
    :param edges_hardclipped: Edges (hard-clipped) GeoDataFrame for route shape matching.
    :return: GeoDataFrame with sequential edges geometries.
    """
    # Create next shape points
    shapes['shape_pt_lat_next'] = shapes['shape_pt_lat'].shift(-1)
    shapes['shape_pt_lon_next'] = shapes['shape_pt_lon'].shift(-1)
    shapes = shapes.dropna(subset=['shape_pt_lat_next', 'shape_pt_lon_next'])

    # Get the nearest edges and add shape-shape bearings
    shapes['orig_edge'] = [*map(tuple, ox.get_nearest_edges(graph_obj_h, shapes.shape_pt_lon, shapes.shape_pt_lat))]
    shapes['dest_edge'] = [*map(tuple, ox.get_nearest_edges(graph_obj_h, shapes.shape_pt_lon_next, shapes.shape_pt_lat_next))]
    shapes['shape_bearing'] = shapes.apply(lambda x: ox.bearing.get_bearing((x.shape_pt_lat, x.shape_pt_lon), (x.shape_pt_lat_next, x.shape_pt_lon_next)), axis = 1)
    
    # Create reversed edges for origin and destination
    # of shape-shape for bearing comparison
    shapes['orig_u'] = shapes['orig_edge'].str[0]
    shapes['orig_v'] = shapes['orig_edge'].str[1]

    shapes['rev_orig_u'] = shapes['orig_edge'].str[1]
    shapes['rev_orig_v'] = shapes['orig_edge'].str[0]

    shapes['dest_u'] = shapes['dest_edge'].str[0]
    shapes['dest_v'] = shapes['dest_edge'].str[1]

    shapes['rev_dest_u'] = shapes['dest_edge'].str[1]
    shapes['rev_dest_v'] = shapes['dest_edge'].str[0]

    shapes['rev_orig_edge'] = [*zip(shapes.rev_orig_u, shapes.rev_orig_v, [0] * len(shapes.rev_orig_v))]
    shapes['rev_dest_edge'] = [*zip(shapes.rev_dest_u, shapes.rev_dest_v, [0] * len(shapes.rev_dest_v))]

    # Merge bearing from OSM data for origin edge and reversed origin edge of shape-shape
    shapes = shapes.merge(edges_hardclipped[['u', 'v', 'bearing']]
                          .rename(columns={'u': 'orig_u', 'v': 'orig_v', 'bearing': 'orig_bearing'}),
                          on=['orig_u', 'orig_v'],
                          how='left')

    shapes = shapes.merge(edges_hardclipped[['u', 'v', 'bearing']]
                          .rename(columns={'u': 'rev_orig_u', 'v': 'rev_orig_v', 'bearing': 'rev_orig_bearing'}),
                          on=['rev_orig_u', 'rev_orig_v'],
                          how='left')

    # Correct origin edge of shape-shape using bearing differences
    shapes['bearing_diff_orig'] = np.vectorize(get_bearing_difference)(shapes.shape_bearing,
                                                                       shapes.orig_bearing)
    shapes['bearing_diff_rev_orig'] = np.vectorize(get_bearing_difference)(shapes.shape_bearing,
                                                                           shapes.rev_orig_bearing)
    shapes['corrected_orig_edge'] = np.where((shapes.bearing_diff_orig <= shapes.bearing_diff_rev_orig),
                                             shapes.orig_edge,
                                             shapes.rev_orig_edge)
    shapes.loc[shapes.bearing_diff_rev_orig.isnull(), 'corrected_orig_edge'] = shapes.orig_edge
    logger.info(f'Orientation for origin edge corrected for {len(shapes):,} shape-shape pairs.')

    # Merge bearing from OSM data for destination edge and reversed destination edge of shape-shape
    shapes = shapes.merge(edges_hardclipped[['u', 'v', 'bearing']]
                          .rename(columns={'u': 'dest_u', 'v': 'dest_v', 'bearing': 'dest_bearing'}),
                          on=['dest_u', 'dest_v'],
                          how='left')

    shapes = shapes.merge(edges_hardclipped[['u', 'v', 'bearing']]
                          .rename(columns={'u': 'rev_dest_u', 'v': 'rev_dest_v', 'bearing': 'rev_dest_bearing'}),
                          on=['rev_dest_u', 'rev_dest_v'],
                          how='left')

    # Correct destination edge of shape-shape using bearing differences
    shapes['bearing_diff_dest'] = np.vectorize(get_bearing_difference)(shapes.shape_bearing,
                                                                       shapes.dest_bearing)
    shapes['bearing_diff_rev_dest'] = np.vectorize(get_bearing_difference)(shapes.shape_bearing,
                                                                           shapes.rev_dest_bearing)
    shapes['corrected_dest_edge'] = np.where((shapes.bearing_diff_dest <= shapes.bearing_diff_rev_dest),
                                             shapes.dest_edge,
                                             shapes.rev_dest_edge)
    shapes.loc[shapes.bearing_diff_rev_dest.isnull(), 'corrected_dest_edge'] = shapes.dest_edge
    logger.info(f'Orientation for destination edge corrected for {len(shapes):,} shape-shape pairs.')

    # Get correct origin and destination edge for shape-shape
    shapes['single_edge'] = 1 * (shapes.corrected_orig_edge == shapes.corrected_dest_edge)
    shapes['corrected_u'] = shapes['corrected_orig_edge'].str[0]
    shapes['corrected_v'] = shapes['corrected_dest_edge'].str[1]

    # Get shortest path for each shape-shape in sequential order and keep unique segments
    od_shapes = shapes[['corrected_u', 'corrected_v']].drop_duplicates()
    od_shapes['order'] = list(range(1, len(od_shapes) + 1))
    od_shapes['u'] = od_shapes.apply(lambda x: ox.shortest_path(G, x.corrected_u, x.corrected_v), axis = 1)

    od_shapes = od_shapes.explode('u')
    od_shapes['v'] = od_shapes.u.shift(-1)
    od_shapes['remove_tag'] = 1 * (od_shapes.order != od_shapes.order.shift(-1))
    od_shapes = od_shapes[od_shapes.remove_tag != 1]
    od_shapes = od_shapes.drop_duplicates(subset=['u', 'v'])
    route_path = construct_path(od_shapes, edges_hardclipped)
    logger.info(f'Route path constructed with {len(route_path):,} edges.')
    return route_path


def project_point_on_line(pt, line_geom):
    """
    Projects a point on route line geometry.
    :param pt: Shapely Point object.
    :param line_geom: Shapely LINESTRING object.
    :return: Projected coordinates.
    """
    return line_geom.interpolate(line_geom.project(pt))


def project_points_on_line(points_gdf, line_geom):
    """
    Returns projected points on a line geometry in EPSG:3857.
    :param points_gdf: Points GeoDataFrame.
    :param line_geom: Shapely LINESTRING object.
    :return: Points GeoDataFrame with projected coordinates in EPSG:3857.
    """
    projected_points_gdf = points_gdf.copy().to_crs(CRS_METER)
    projected_points_gdf['projected_pt'] = (projected_points_gdf['geometry']
                                            .apply(lambda x: project_point_on_line(x, line_geom)))
    projected_points_gdf['geometry'] = projected_points_gdf['projected_pt']
    return projected_points_gdf.drop(columns=['projected_pt'])


def merge_stop_sequences(route_stop_seq, route_path):
    """
    Merge stop sequence information.
    :param route_stop_seq: Stop sequence GeoDataFrame for candidate route pattern.
    :param route_path: GeoDataFrame with sequential edges geometries.
    :return: GeoDataFrame with sequential edges geometries and stop sequence information.
    """
    # Project stops to unified route line geometries.
    route_stop_seq = project_points_on_line(route_stop_seq, route_path.dissolve().geometry[0])

    # Use projected stop coordinates to spatially join OSM edges.
    route_path_buffer = route_path.copy()
    route_path_buffer['geometry'] = route_path.buffer(5)
    route_stop_seq_with_path = gpd.sjoin(route_stop_seq,
                                         route_path_buffer,
                                         how='left',
                                         predicate='within').drop_duplicates('stop_sequence')

    # Calculate distance of stop location from the start point of corresponding matched edges.
    stop_geo = route_stop_seq_with_path['geometry']
    edge_geo = (route_stop_seq_with_path[['edge_code']]
                .merge(route_path[['edge_code', 'geometry']])
                .drop_duplicates('edge_code')['geometry'])
    route_stop_seq_with_path['dist_from_edge_start_ft'] = [e.project(s) * 3.28084 for s, e in zip(stop_geo, edge_geo)]

    # Keep select columns and merge matched stops information.
    sel_cols = ['stop_sequence', 'stopid', 'pattern_id', 'edge_code', 'edge_len_ft', 'dist_from_edge_start_ft']
    route_path_with_stops = route_path.merge(route_stop_seq_with_path[sel_cols],
                                             on=['edge_code', 'edge_len_ft'],
                                             how='left')
    logger.info(f'{len(route_stop_seq):,} stops merged with route edges.')
    return route_path_with_stops


def process_route_edges_outputs(route_path_with_stops, route_stop_seq, route_pattern_id, crs = CRS):
    """
    Create additional columns and do final processing of matched network links.
    :param route_path_with_stops: Route path (constructed with OSM edges) GeoDataFrame with stop information.
    :param route_stop_seq: Stop sequence GeoDataFrame for candidate route pattern.
    :param route_pattern_id: Candidate route pattern ID in RRDD format (RR = Route Number, DD = Direction).
    :param crs: Coordinate Reference System in "EPSG:XXXX" format.
    :return: Final network links GeoDataFrame with necessary information.
    """
    route_path_with_stops['from_stop_seq'] = np.where(route_path_with_stops['stop_sequence'].isna(),
                                                      np.nan,
                                                      route_path_with_stops['stop_sequence'])
    route_path_with_stops['from_stop_seq'] = (route_path_with_stops['from_stop_seq']
                                              .fillna(method='ffill')
                                              .astype(int))

    route_path_with_stops['to_stop_seq'] = route_path_with_stops['from_stop_seq'] + 1

    route_path_with_stops.loc[route_path_with_stops['edge_order'].index.max():, 'from_stop_seq'] = \
        route_path_with_stops['from_stop_seq'] - 1
    route_path_with_stops.loc[route_path_with_stops['edge_order'].index.max():, 'to_stop_seq'] = \
        route_path_with_stops['to_stop_seq'] - 1

    route_path_with_stops = (route_path_with_stops
                             .merge(route_stop_seq.rename(columns={'stopid': 'from_stopid',
                                                                   'stop_sequence': 'from_stop_seq'})[
                                        ['from_stopid', 'from_stop_seq']])
                             .merge(route_stop_seq.rename(columns={'stopid': 'to_stopid',
                                                                   'stop_sequence': 'to_stop_seq'})[
                                        ['to_stopid', 'to_stop_seq']]))

    route_path_with_stops['from_node'] = route_path_with_stops['edge_code'].str.split('-', expand=True)[0]
    route_path_with_stops['to_node'] = route_path_with_stops['edge_code'].str.split('-', expand=True)[1]
    route_path_with_stops['link_code'] = (route_path_with_stops['from_stopid'].astype(str) +
                                          '-' + route_path_with_stops['to_stopid'].astype(str))
    route_path_with_stops['pattern_id'] = route_pattern_id

    final_route_edges = route_path_with_stops[['pattern_id', 'stop_sequence', 'stopid',
                                               'edge_order', 'edge_code', 'edge_len_ft',
                                               'from_node', 'to_node', 'geometry']].to_crs(crs)
    logger.info(f'Final processing completed. {len(final_route_edges):,} matched route edges returned.')
    if len(final_route_edges) != final_route_edges.edge_code.nunique():
        logger.warning(f'Route path contains repeated edges. '
                       f'Expected {final_route_edges.edge_code.nunique()}, returned {len(final_route_edges)}.')

    # Check matching results
    n, repeated_edges, msg = check_matching_results(final_route_edges)
    if n > 0:
        logger.warning(f'{n} edges have repetitions. Edges are {repeated_edges}.')
    return final_route_edges, n, repeated_edges


def check_matching_results(gdf):
    """
    Check for repeated edges in the final matched edges GeoDataFrame.
    :param gdf: Final network links GeoDataFrame with necessary information.
    :return: Number of repeated edges, Repeated edge codes, custom messages for logging.
    """
    n = 0
    msg = 'No repetition.'
    repeated_edges = (gdf[gdf.groupby('edge_code')['edge_code']
                          .transform('size') > 1].edge_code.unique().tolist())
    if len(repeated_edges) > 0:
        n = len(repeated_edges)
        msg = f'{n} edges have repetitions. Edges are {repeated_edges}.'
    return n, repeated_edges, msg


def export_gdf(gdf, file_name, output_path, crs=CRS):
    """
    Export a GeoDataFrame.
    :param gdf: GeoDataFrame to be exported.
    :param file_name: Name of the file to be exported.
    :param output_path: Output path (folder) directory.
    :param crs: Coordinate Reference System in "EPSG:XXXX" format.
    :return: None.
    """
    gdf.to_crs(crs).to_file(os.path.join(output_path, file_name))
    logger.info(f'{file_name} exported to {output_path}.')


def export_final_network_edges(output_dir, gdf, route_pattern_id, crs=CRS, overwrite=True):
    """
    Export final processed network links in .geojson format.
    :param output_dir: Parent output path (folder) directory.
    :param gdf: Final network links GeoDataFrame to be exported.
    :param route_pattern_id: Candidate route pattern ID in RRDD format (RR = Route Number, DD = Direction).
    :param crs: Coordinate Reference System in "EPSG:XXXX" format.
    :param overwrite: If True, file is overwritten if another file already exists.
    :return: None.
    """
    os.makedirs(os.path.join(output_dir, route_pattern_id), exist_ok=True)
    output_path = os.path.join(output_dir, route_pattern_id)
    if overwrite is False:
        file_name = route_pattern_id + f'_FINAL_BUS_NET_LINK_{get_today()}.geojson'
    else:
        file_name = route_pattern_id + f'_FINAL_BUS_NET_LINK.geojson'
    export_gdf(gdf, file_name, output_path, crs)
