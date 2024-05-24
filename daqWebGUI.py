import plotly.graph_objs as go
import plotly as py
import dash
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html
import base64
import h5py

import cProfile

import time
from time import sleep

from numpy import exp, random, arange, outer, sin, pi

import os
import sys
import time
import numpy as np
#from multiprocessing import Process, Queue
import re
#import sis3316 as sis
#import pylab
#import csv
import json
import atexit
import traceback
import argparse
import ast

TESTING_GUI = False
TESTING_DAQ = False
FROM_FILE = False
from_file_name = ''

NDETECTORS = 128
ENERGY_RANGE = (0,1000)
DETECTOR_INDEX = {}

START_TIME = 0
DAQ_STARTED = False

#data_fields = ['format', 'channel', 'header', 'timestamp', 'adc_max', 'adc_argmax', 'gate1', 'pileup',
#               'repileup','gate2', 'gate3', 'gate4', 'gate5', 'gate6', 'gate7', 'gate8', 'maw_max', 'maw_after_trig',
#               'maw_before_trig', 'en_start', 'en_max', 'raw_data', 'maw_data']

data_fields = ['channel', 'timestamp']

graph_data_fields = ['counts', 'heatmap', 'energy']

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

external_css = ["https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/css/materialize.min.css"]

external_js = ['https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/js/materialize.min.js']

raw_length = 300
raw_max = 9000
raw_min = 8000

def get_file_data(n,temp_data):
    datafile = h5py.File(from_file_name,'r')
    #idx = n % len(datafile['raw_data'])
    idx = n % len(datafile['event_data'])
    total_data = {}
    detector_data = {}
    detector = int(datafile['event_data']['det'][idx])
    print("Getting data for detector",detector)
    event_data = np.array(datafile['event_data'])
    if temp_data:
        old_data = json.loads(temp_data)
        try:
            energies = np.array(event_data['gate2']) - 2*np.array(event_data['gate1'])
            all_energies = np.concatenate((np.array(old_data[str(detector)]['energy']),energies))
            detector_data['energy'] = all_energies.tolist()
        except:
            detector_data['energy'] = (np.array(event_data['gate2']) - 2*np.array(event_data['gate1'])).tolist()
    else:
        detector_data['energy'] = (np.array(event_data['gate2']) - 2*np.array(event_data['gate1'])).tolist()
    detector_data['timestamp'] = datafile['event_data']['timestamp'][:idx].tolist()
    total_data[detector] = detector_data
    return json.dumps(total_data)

def create_histogram(data, bins, range=(0,2000)):
    # Preprocess data
    bin_edges = np.linspace(range[0], range[1], bins + 1)
    
    # Convert data into integer indices
    indices = np.searchsorted(bin_edges, data, side='right') - 1
    
    # Create initial histogram
    histogram = np.bincount(indices, minlength=bins)
    
    return histogram

def update_histogram(histogram, new_data, range=(0,2000)):
    # Convert new data into integer indices
    bin_edges = np.linspace(range[0], range[1], len(histogram) + 1)
    indices = np.searchsorted(bin_edges, new_data, side='right') - 1
    
    # Count occurrences of each index
    counts = np.bincount(indices, minlength=len(histogram))
    # Ensure histogram and counts have the same shape
    if histogram.shape != counts.shape:
        histogram.resize(counts.shape)
    # Update the histogram
    histogram += counts
    
    return histogram

def start_daq(file_list,arg_list):
    if DAQ_STARTED:
        return
    print("Starting data acquisition")
    if TESTING_GUI:
        cmd = "echo 'testing GUI'"
    elif FROM_FILE:
        cmd = "echo 'plotting data from input file: {}'".format(from_file_name)
    else:
        file_args = None
        if file_list is not None:
            file_args = ' '.join([file for file in file_list])
            script = file_list[0]
        args = None
        if arg_list is not None:
            args = ' '.join([str(arg) for arg in arg_list])

        if TESTING_DAQ:
            #cmd = "python3 data_subscriber.py --files {} --ips {} --test --gui > daq.log 2>&1".format(file_args,ip_args)
            cmd = "python {} {} --test --gui > daq.log 2>&1".format(script,args)
        else:
            #cmd = "python3 data_subscriber.py --files {} --ips {} --gui > daq.log 2>&1".format(file_args,ip_args)
            cmd = "python {} {} --gui > daq.log 2>&1".format(script,args)

        send_queue_cmd('START')

    print(cmd)
    os.system(cmd)
    DAQ_STARTED = True
    # Send command to command-line to start daq

def stop_daq():
    print("Stopping data acquisition")
    if not TESTING_GUI and not FROM_FILE:
        send_queue_cmd('EXIT')
    DAQ_STARTED = False


def send_queue_cmd(cmd):
    '''
    Send commands for sensor DAQs
        - valid commands: START, STOP, EXIT
    '''
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='fromGUI')
    print("Sending cmd: {}".format(cmd))
    message = {'id': 0, 'cmd': cmd}
    channel.basic_publish(exchange='',
                          routing_key='fromGUI',
                          body=json.dumps(message))
    connection.close()

def receive_queue_data():
    '''
    Receive data from sensor DAQs
    '''
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='toGUI')
    method_frame, header_frame, body = channel.basic_get(queue='toGUI')
    if body is not None:
        # message from d3s is coming back as bytes
        if type(body) is bytes:
            body = body.decode("utf-8")
        message = json.loads(body)
        channel.basic_ack(delivery_tag=method_frame.delivery_tag)
        connection.close()
        return message
    else:
        connection.close()
        return None

def clear_queue():
    print("Initializing queues... clearing out old data")
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='toGUI')
    channel.queue_delete(queue='toGUI')
    connection.close()

    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='fromGUI')
    channel.queue_delete(queue='fromGUI')
    connection.close()


app = dash.Dash('SIS-daq',
#                external_scripts=external_js,
                external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)

app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

app.layout = html.Div([
    html.Div([
        html.H4('SIS Data Acquisition Interface',
                style={'float': 'left',})
        ],className="row"),
    html.Div([
        html.P('Provide the DAQ script and arguments before starting',
                style={'float': 'left',}),
        ],className="row"),
    html.Div([
        html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    html.A('Select DAQ Script')
                ]),
                style={
                    'width': '30%',
                    #'height': '50px',
                    #'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    #'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True)
                #]),
            ]),#,className="row"),
        html.Div(dcc.Input(id='input-box', type='text', 
                placeholder='Enter DAQ arguments ...',
                style={
                    'width': '30%',
                    #'height': '50px',
                    #'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    #'margin': '10px'
                },
            )),
        ],className='row'),
    #html.Div(id='output-data-upload'),
    #html.Br(),
    html.Div([
        html.Button('Start', id='start_button', style={'color': 'darkblue'}),
        html.Button('Stop', id='stop_button', style={'color': 'red'}),
        html.Div(dcc.Input(id='detector-select', type='text', 
                placeholder='Select detector: [9,1]',
            )),
        ],className='row'),
    dcc.Dropdown(id='graph-types',
                 options=[{'label': s, 'value': s}
                          for s in graph_data_fields],
                 value=['heatmap','counts'],
                 multi=True
                 ),
    html.Div(children=html.Div(id='graphs'), className='row'),
    dcc.Interval(
        id='graph-update',
        interval=1*1000,
        n_intervals=0),
    dcc.Interval(
        id='data-update',
        interval=1*1000,
        n_intervals=0),

    # Hidden div inside the app that stores the intermediate values for graphing
    html.Div(id='intermediate-values', style={'display': 'none'})

    ],className="container",style={'width':'98%','margin-left':10,'margin-right':10,'max-width':50000})


@app.callback(Output('output-container-button', 'children'),
              [Input('start_button', 'n_clicks'),
               Input('stop_button', 'n_clicks')],
              [State('input-box', 'value'),
               State('upload-data', 'filename'),
               State('upload-data', 'contents')])
def update_output(start_clicks, stop_clicks, values, list_of_names, list_of_contents):
    print("start_clicks: {}, stop_clicks: {}, values: {}, list_of_names: {}".format(start_clicks, stop_clicks, [values], list_of_names))
    # store config files in temporary files in the cwd to avoid need for absolute paths
    if list_of_contents is not None and start_clicks == 1 and stop_clicks is None:
        for filename, contents in zip(list_of_names, list_of_contents):
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            decoded = str(decoded, 'utf-8')

            with open(filename,'w') as config:
                content_dict = json.loads(decoded)
                json.dump(content_dict,config)

    if start_clicks is not None:
        print("Starting {} time".format(start_clicks))
        if stop_clicks is not None:
            print("Starts: {}, Stops: {}".format(start_clicks, stop_clicks))
            if start_clicks > stop_clicks:
                start_daq(list_of_names,[values])
                return 'Starting DAQ with config files: "{}" going to IPs: {}'.format(
                    list_of_names,
                    values
                )
            else:
                print("Stopping {} times after {} starts".format(stop_clicks, start_clicks))
                stop_daq()
                return 'Stopping DAQ'
        else:
                start_daq(list_of_names,[values])
                return 'Starting DAQ with config files: "{}" going to IPs: {}'.format(
                    list_of_names,
                    values
                )
    else:
        return 'Click Start to run the DAQ with config files: "{}" going to IPs: {}'.format(
                        list_of_names,
                        values
                    )


@app.callback(Output('intermediate-values','children'),
             [Input('data-update','n_intervals')],
             [State('intermediate-values','children'),
              State('start_button', 'n_clicks')])
def update_data(n,temp_data,start_clicks):
    if start_clicks is None:
        return None
    else:
        if TESTING_GUI:
            dets = np.random.randint(0,NDETECTORS,12)
            for idet in dets:
                fake_data = make_test_data()
                temp_data = add_data(str(idet),fake_data,temp_data)
            data = temp_data
        elif FROM_FILE:
            data = get_file_data(n,temp_data)
        else:
            message = receive_queue_data()
            while message is not None:
                temp_data = add_data(message['id'],message['data'],temp_data)
                #print("receive_queue_data: {}".format(message))
                message = receive_queue_data()
            data = temp_data
        return data

def make_test_data():
    fake_data = {}
    fake_data['timestamp'] = time.time() -START_TIME
    fake_data['energy'] = np.random.exponential(scale=200,size=int(25*np.random.random()))
    return fake_data

def get_energy_data(new_data,old_data):
    #new_energies = np.array(new_data['gate2']) - 2*np.array(new_data['gate1'])
    new_energies = np.array(new_data['energy'])
    counts = len(new_energies)
    if not old_data:
        histo_data = create_histogram(new_energies,250,ENERGY_RANGE)
    else:
        histo_data = update_histogram(np.array(old_data), new_energies, ENERGY_RANGE)
    data = histo_data.tolist()
    return data, counts

def get_channel_data(new_data,old_data):
    this_channel = new_data['channel']
    if old_data is None:
        data = [this_channel]
    else:
        data = old_data
        data.append(this_channel)
    return data

def add_data(detector, data, temp_data):
    if not temp_data:
        total_data = {}
        updated_data = {}
        updated_data['timestamp'] = [data['timestamp']]
        updated_data['energy'], counts = get_energy_data(data,None)
        updated_data['counts'] = [counts]
    else:
        total_data = json.loads(temp_data)
        if detector in total_data.keys():
            #print("add_data: {}".format(total_data))
            updated_data = total_data[detector]
            updated_data['energy'], counts = get_energy_data(data, updated_data['energy'])
            updated_data['counts'].append(counts)
            updated_data['timestamp'].append(data['timestamp'])
            if data['timestamp'] > 100:
                updated_data['counts'].pop(0)
                updated_data['timestamp'].pop(0)
        else:
            updated_data = {}
            updated_data['energy'], counts = get_energy_data(data,None)
            updated_data['timestamp'] = [data['timestamp']]
            updated_data['counts'] = [counts]
 
    total_data[detector] = updated_data
    #print("Combined data counts for",detector,":",total_data[detector]['counts'])
    try:
        json.dumps(total_data)
    except Exception as e:
        print(e)
        pass
    return json.dumps(total_data)


def read_in_file(info_file, folder_location):
    with open(folder_location+info_file) as f:
        file_data = f.readlines()
    for i, info in enumerate(file_data):
        file_data[i] = info.strip('\n')
        
    return np.array(file_data)

def get_det_index():
    detector_positions = {}
    user_path = os.getenv("HOME")
    file_path = ""
    if sys.platform == 'win32':
        file_path = user_path + '\\CAMIS\\Data-Files\\Basic_System_Info\\'
    else:
        file_path = user_path + "/CAMIS/Data-Files/Basic_System_Info/"
    channel_order = read_in_file("channel_order.txt",file_path)
    detector_position_index = read_in_file("detector_position_indexes.txt",file_path)
    #channel_index = np.where(channel_order==idet)[0][0]
    for i,channel in enumerate(channel_order):
        channel_position = int(detector_position_index[i])
        ic = channel_position%24
        ir = int(channel_position/24)
        detector_positions[channel] = [ir,ic]
    return detector_positions

'''
def get_det_index(idet):
    channel_order = read_in_file("channel_order.txt","~/CAMIS/Data-Files/Basic_System_Info/")
    detector_position_index = read_in_file("detector_position_indexes.txt","~/CAMIS/Data-Files/Basic_System_Info/")
    channel_position = np.where(channel_order==idet)[0][0]
    ic = channel_position%24
    ir = int(channel_position/24)
    return ir,ic
'''

def get_current_data(graph_type, alldata_string):
    times = []
    data = []
    if alldata_string is not None:
        #print("Getting current data: {} of length {}".format(alldata_string,len(alldata_string)))
        if len(alldata_string) > 0:
            alldata = json.loads(alldata_string)
            if graph_type=='heatmap':
                data = np.empty([10,24])
                data.fill(None)
                try:
                    for idet in alldata:
                        count = alldata[idet]['counts'][-1]
                        ir,ic = DETECTOR_INDEX[idet]
                        data[ir,ic] = count
                except:
                    print("error with 2D data!")
                    pass
            else:
                for id in alldata:
                    #print(alldata[id])
                    #print("")
                    data.append(alldata[id][graph_type])
                    times.append(alldata[id]['timestamp'])
        else:
            times.append([0])
            data.append([0])
    else:
        times.append([0])
        data.append([0])
    return times, data

def get_det_id(pos):
    return list(DETECTOR_INDEX.keys())[list(DETECTOR_INDEX.values()).index(pos)]

def make_graph(graph_name, times, data, detector_pos):
    if graph_name=='counts':
        traces = list()
        for idet,itimes in enumerate(times):
            igraph = go.Scatter(x=itimes,y=data[idet],name=idet)
            traces.append(igraph)
        layout = {
                  'margin': {'l': 25, 'r': 5, 't': 25, 'b': 20},
                  #'margin':{'l':50,'r':10,'t':45,'b':30},
                  'title': {
                      'text': 'Counts vs time',
                      'y': 0.95,  # Adjust the vertical position of the title
                      'x': 0.5,
                      'xanchor': 'center',
                      'yanchor': 'top'
                  },
                  'uirevision':"stay",
                  'xaxis':{'autorange':True},
                  'yaxis':{'autorange':True}
                  }

    if graph_name=='energy':
        traces = list()
        if(detector_pos):
            idet = get_det_id(detector_pos)
        else:
            idet = get_det_id([9,1])
        #for idet,itimes in enumerate(times):
        #histo_data, bins = np.histogram(data[i],500)
        histo_data = data[int(idet)]
        bins = np.linspace(ENERGY_RANGE[0], ENERGY_RANGE[1], len(histo_data))+.5
        ihisto = go.Bar(x=bins,y=histo_data,name=idet)
        traces.append(ihisto)
        layout = {'barmode':'overlay',
                  'bargap':0.0,
                  #'margin':{'l':50,'r':10,'t':45,'b':30},
                  'margin': {'l': 25, 'r': 5, 't': 25, 'b': 15},
                  'title':'{}'.format('Energy'),
                  'uirevision':"stay",
                  'xaxis':{'autorange':True},
                  'yaxis':{'autorange':True,'type': 'log'}}

    if graph_name=='heatmap':
        traces = list()
        data_array = np.array(data)
        if len(data_array.shape)>1:
            specific_values = [0.0, -1.0]
            specific_colors = ['#FF0000', '#000000']  # Red and Green
            # Calculate the marker size based on the heatmap dimensions
            marker_size = int( 600 / max(data_array.shape) )

            ihisto = go.Heatmap(z=data_array, 
                                colorscale = 'Plasma',
                                colorbar=dict(
                                    lenmode='fraction',
                                    len=0.85,
                                    thickness=15,
                                    x=1.02,
                                    y=0.5,
                                    yanchor='middle'
                                ),
                                )
            mask_data = np.nan_to_num(data_array,nan=-1.0)
            mask = np.isin(mask_data, specific_values)
            scatter_trace = go.Scatter(
                x=[j for i in range(mask_data.shape[0]) for j in range(mask_data.shape[1]) if mask[i][j]],
                y=[i for i in range(mask_data.shape[0]) for j in range(mask_data.shape[1]) if mask[i][j]],
                mode='markers',
                marker=dict(
                    color=[specific_colors[specific_values.index(mask_data[i][j])] for i in range(mask_data.shape[0]) for j in range(mask_data.shape[1]) if mask[i][j]],
                    size=marker_size,
                    sizeref=marker_size,
                    sizemode='diameter',
                    symbol='square',
                    line=dict(width=0)
                ),
                hoverinfo='none'  # Disable hover information
            )
            traces.append(ihisto)
            traces.append(scatter_trace)

        layout = {
            'margin': {'l': 25, 'r': 5, 't': 25, 'b': 15},
            'title': {
                'text': 'Detector Heatmap',
                'y': 0.95,  # Adjust the vertical position of the title
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },

            'uirevision': "stay",
            'xaxis': {
                      'constrain': 'domain',
                      'showgrid': False,
                      'showline': False,
                      'zeroline': False
                },
            'yaxis': {
                      'scaleanchor': 'x', 
                      'constrain': 'domain',
                      'showgrid': False,
                      'showline': False,
                      'zeroline': False
                },
            'plot_bgcolor': 'rgb(0,0,0)',
            'aspectmode': 'manual',
            'aspectratio': {'x': data_array.shape[1], 'y': data_array.shape[0]}
        }

    return traces, layout

@app.callback(Output('graphs','children'),
             [Input('graph-types', 'value')],
             [State('intermediate-values','children'),
              State('detector-select', 'value')])
def update_graphs(graph_names, current_data, detector_id):

    graphs = []
    if detector_id:
        detector_id = ast.literal_eval(detector_id)
    for graph_name in graph_names:
        times, graph_data = get_current_data(graph_name,current_data)
        traces, layout = make_graph(graph_name,times,graph_data,detector_id)

        graphs.append(html.Div(dcc.Graph(
            id=graph_name,
            animate=False,
            style={'width': '100%', 'height': '80vh'},
            figure={'data': traces,'layout' : layout}
            )))


    return graphs

@app.callback(Output('counts','figure'),
             [Input('graph-update', 'n_intervals')],
             [State('intermediate-values','children'),
              State('start_button', 'n_clicks'),
              State('detector-select', 'value')])
def update_counts_data(n, current_data, n_clicks, detector_id):
    if n_clicks is not None:
        if detector_id:
            detector_id = ast.literal_eval(detector_id)
        times, graph_data = get_current_data('counts',current_data)
        traces, layout = make_graph('counts',times,graph_data,detector_id)
        return {'data': traces, 'layout': layout}
    else:
        return dash.no_update

@app.callback(Output('energy','figure'),
             [Input('graph-update', 'n_intervals')],
             [State('intermediate-values','children'),
              State('start_button', 'n_clicks'),
              State('detector-select', 'value')])
def update_energy_data(n, current_data, n_clicks, detector_id):
    if n_clicks is not None:
        if detector_id:
            detector_id = ast.literal_eval(detector_id)
        times, graph_data = get_current_data('energy',current_data)
        traces, layout = make_graph('energy',times,graph_data, detector_id)
        return {'data': traces, 'layout': layout}
    else:
        return dash.no_update

@app.callback(Output('heatmap','figure'),
             [Input('graph-update', 'n_intervals')],
             [State('intermediate-values','children'),
              State('start_button', 'n_clicks'),
              State('detector-select', 'value')])
def update_heatmap_data(n, current_data, n_clicks, detector_id):
    if n_clicks is not None:
        if detector_id:
            detector_id = ast.literal_eval(detector_id)
        times, graph_data = get_current_data('heatmap',current_data)
        traces, layout = make_graph('heatmap',times,graph_data, detector_id)
        return {'data': traces, 'layout': layout}
    else:
        return dash.no_update

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_gui", "-g",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--test_daq", "-t",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--from_file", "-f",
        help='provide data from file to be displayed',
        default=None,
    )

    args = parser.parse_args()
    arg_dict = vars(args)
    TESTING_GUI = arg_dict['test_gui']
    TESTING_DAQ = arg_dict['test_daq']
    DETECTOR_INDEX = get_det_index()
    START_TIME = time.time()
    print("DETECTOR_INDEX:",DETECTOR_INDEX)
    if args.from_file:
        FROM_FILE = True
        from_file_name = args.from_file

    if not TESTING_GUI and not FROM_FILE:
        import pika

    try:
        if not TESTING_GUI and not FROM_FILE:
            clear_queue()
        app.run_server(debug=False)
    except:
        if not TESTING_GUI and not FROM_FILE:
            send_queue_cmd('EXIT')
        # Still want to see traceback for debugging
        print('ERROR: GUI quit unexpectedly!')
        traceback.print_exc()
        pass
