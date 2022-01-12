# This script is used to run route shape matching processes
# for multiple route patterns using loop.

# Initialization
################
# Import standard and local packages
import os
import sys
import logging
import warnings
import pandas as pd
import osmnx as ox
import numpy as np

sys.path.append('../src/')
import route_shape_matching.rm_utilities as rmu


# Session settings
warnings.filterwarnings('ignore')

# Configure logger
logging.basicConfig(level=logging.CRITICAL,
                    format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('WMATA')
logger.setLevel(logging.INFO)

# Directory and parameters
##########################
# - Set directory to read the geospatial files.
# - Set or import parameters from local modules such as coordinate reference system.
data_dir = os.path.normpath(r'..\\data')
os.makedirs(os.path.join(data_dir, 'processed', 'route_shapes'), exist_ok=True)
output_dir = os.path.join(data_dir, 'processed', 'route_shapes')

# Parameters
CRS = rmu.CRS
CRS_METER = rmu.CRS_METER

# GIS route line geometries
gis_lines_geojson = os.path.join(data_dir,
                                 'external',
                                 'gis_bsrt_line',
                                 'gis_bsrt_line.geojson')

gtfs_shapes_text = os.path.join(data_dir,
                                'external',
                                'gtfs_20210903',
                                'shapes.txt')

stop_seq_csv = os.path.join(data_dir,
                            'external',
                            'stop_sequence',
                            'bus_net_stop_sequence.csv')

# Load OSM graph
################
# - Read OSM network as graph object.
# - Extract edges and nodes with their spatial indexes for faster queries.
G = ox.load_graphml(os.path.join(data_dir,
                                 'processed',
                                 'osm_graph',
                                 '20220104_osm_graph.graphml'))

# Nodes and edges and their spatial indexes from graph
nodes, edges, node_sindex, edge_sindex = rmu.get_graph_nodes_edges_sindex(G)


# Read data
###########
# - Read all GIS route lines.
# - Read all GTFS shapes.
# - Read all stop sequences.
gis_routes = rmu.read_all_gis_route_lines(gis_lines_geojson, CRS)
all_shapes = rmu.read_gtfs_shapes(gtfs_shapes_text, CRS)
all_stop_seq = rmu.read_all_routes_stop_sequences(stop_seq_csv, CRS)


# Selection parameters
######################
# route_pattern_list = all_shapes.shape_id.unique().tolist()
route_pattern_list = ['G801']
query_month = 20210903
status_code = []
stop_seq_num = []
shape_pt_num = []
final_edge_num = []
repeated_edges_num = []
repeated_edges_list = []

for i, route_pattern_id in enumerate(route_pattern_list, start=1):
    logger.info(f'({i}/{len(route_pattern_list)}) Processing for {route_pattern_id}... ')
    try:
        gis_route = rmu.get_single_gis_route_line(gis_routes, route_pattern_id, query_month)
        shapes = rmu.get_single_route_gtfs_shapes(all_shapes, route_pattern_id)
        shapes_line = rmu.get_line_geom_from_shapes(shapes, crs=CRS)
        route_stop_seq = rmu.get_single_route_stop_sequences(all_stop_seq, route_pattern_id)

        # Create buffers for GTFS lines and GIS lines
        shapes_line_buffer = rmu.create_buffer(shapes_line,
                                               buffer_in_meter=25,
                                               output_crs=CRS,
                                               sort_columns=['shape_id'])

        gis_route_buffer = rmu.create_buffer(gis_route,
                                             buffer_in_meter=25,
                                             output_crs=CRS,
                                             sort_columns=['shape_id'])[['shape_id', 'geometry']]
        buffer_geom = rmu.create_union_of_buffer_geom(shapes_line_buffer, gis_route_buffer, only_geom=True)
        g_h, ge_h = rmu.get_hardclipped_edges(G, shapes, edges, buffer_geom, return_nodes=False)
        route_path = rmu.get_route_path(shapes, g_h, ge_h)
        route_path_with_stops = rmu.merge_stop_sequences(route_stop_seq, route_path)
        final_route_edges, n, repeated_edges = rmu.process_route_edges_outputs(route_path_with_stops, route_stop_seq,
                                                                               route_pattern_id, crs=CRS)
        rmu.export_final_network_edges(output_dir, final_route_edges, route_pattern_id, overwrite=True)
        status_code.append('Completed')
        stop_seq_num.append(len(route_stop_seq))
        shape_pt_num.append(len(shapes))
        final_edge_num.append(len(final_route_edges))
        repeated_edges_num.append(n)
        repeated_edges_list.append(repeated_edges)

    except:
        status_code.append('Error/Not Completed')
        stop_seq_num.append(np.nan)
        shape_pt_num.append(np.nan)
        final_edge_num.append(np.nan)
        repeated_edges_num.append(np.nan)
        repeated_edges_list.append(np.nan)

# Create DataFrame with results
results = pd.DataFrame({'pattern_id': route_pattern_list,
                        'status': status_code,
                        'stop_count': stop_seq_num,
                        'shape_count': shape_pt_num,
                        'edge_count': final_edge_num,
                        'repeated_edges_count': repeated_edges_num,
                        'repeated_edges': repeated_edges_list})

# Export results DataFrame
results.to_csv(os.path.join(data_dir, 'processed', f'{rmu.get_today()}_RESULTS.csv'), index = False)
