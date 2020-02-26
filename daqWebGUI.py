import plotly.graph_objs as go
import plotly as py
import dash
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html

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
import pika
import json
import atexit
import traceback
import argparse

TESTING = True

#data_fields = ['format', 'channel', 'header', 'timestamp', 'adc_max', 'adc_argmax', 'gate1', 'pileup',
#               'repileup','gate2', 'gate3', 'gate4', 'gate5', 'gate6', 'gate7', 'gate8', 'maw_max', 'maw_after_trig',
#               'maw_before_trig', 'en_start', 'en_max', 'raw_data', 'maw_data']

data_fields = ['channel', 'timestamp', 'raw_data']

graph_data_fields = ['raw_data', 'channel']

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

external_css = ["https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/css/materialize.min.css"]

external_js = ['https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/js/materialize.min.js']

clear_click_counter = 0

def start_daq():
    print("Starting data acquisition")
    # Send command to command-line to start daq

def stop_daq():
    print("Stopping data acquisition")
    send_queue_cmd('EXIT')


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

app.layout = html.Div([
    html.Div([
        html.H4('SIS Data Acquisition Interface',
                style={'float': 'left',})
        ],className="row"),
    html.Div([ 
        html.P('Provide the DAQ config file(s) and acquisition system IP addresses before starting',
                style={'float': 'left',}),
        ],className="row"),
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                html.A('Select Config Files')
            ]),
            style={
                'width': '30%',
#                'height': '50px',
#                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
#                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=True)
#            ]),
        ],className="row"),
    html.Div(id='output-data-upload'),
    html.Br(),
    html.Div(dcc.Input(id='input-box', type='text', placeholder='Enter DAQ IPs ...')),
    html.Div([
        html.Button('Start', id='start_button', style={'color': 'blue'}),
        html.Button('Clear', id='clear_button', style={'color': 'green'}),
        html.Button('Stop', id='stop_button', style={'color': 'red'}),
        ],className='row'),
    html.Div(id='output-container-button',
             children='Enter a value and press submit'),
    dcc.Dropdown(id='graph-types',
                 options=[{'label': s, 'value': s}
                          for s in graph_data_fields],
                 value=['raw_data'],
                 multi=True
                 ),
    html.Div(children=html.Div(id='graphs'), className='row'),
    dcc.Interval(
        id='graph-update',
        interval=1*2000,
        n_intervals=0),
    dcc.Interval(
        id='data-update',
        interval=1*2000,
        n_intervals=0),

    # Hidden div inside the app that stores the intermediate values for graphing
    html.Div(id='intermediate-values', style={'display': 'none'})

],className="container",style={'width':'98%','margin-left':10,'margin-right':10,'max-width':50000})


@app.callback(Output('output-container-button', 'children'),
              [Input('start_button', 'n_clicks'),
               Input('stop_button', 'n_clicks')],
              [State('input-box', 'value'),
               State('upload-data', 'filename')])
def update_output(start_clicks, stop_clicks, value, list_of_names):
    if start_clicks is not None:
        if stop_clicks is not None:
            if start_clicks > stop_clicks:
                start_daq()
                return 'Starting DAQ with config files: "{}" going to IPs: {}'.format(
                    list_of_names,
                    value
                )
            else:
                stop_daq()
                return 'Stopping DAQ'
        else:
                start_daq()
                return 'Starting DAQ with config files: "{}" going to IPs: {}'.format(
                    list_of_names,
                    value
                )
    return 'Click Start to run the DAQ with config files: "{}" going to IPs: {}'.format(
                    list_of_names,
                    value
                )

@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def update_output(list_of_contents, list_of_names):
    return list_of_names

@app.callback(Output('intermediate-values','children'),
             [Input('data-update','n_intervals'),
              Input('clear_button', 'n_clicks')],
             [State('intermediate-values','children'),
              State('start_button', 'n_clicks')])
def update_data(n,clear_clicks,temp_data,start_clicks):
    global clear_click_counter
    if start_clicks is None:
        return None
    else:
        if clear_clicks is not None:
            if clear_clicks > clear_click_counter:
                clear_click_counter = clear_clicks
                return None

        if TESTING:
            fake_data = make_test_data()
            data = add_data('0',fake_data,temp_data)
        else:
            message = receive_queue_data()
            while message is not None:
                temp_data = add_data(message['id'],message['data'],temp_data)
                message = receive_queue_data()
            data = temp_data
        return data


def make_test_data():
    fake_data = {}
    for data_type in data_fields:
        if data_type=='timestamp':
            fake_data[data_type] = [time.time(),time.time()+.5]
        fake_data[data_type] = list(np.random.random(2))
    return fake_data


def add_data(data_name, data, temp_data):
    if not temp_data:
        total_data = {}
        update_data = {}
        for data_type in data:
            update_data[data_type] = data[data_type]
    else:
        total_data = json.loads(temp_data)
        update_data = total_data[data_name]
        for data_type in data:
            update_data[data_type].extend(data[data_type])

    total_data[data_name] = update_data
    return json.dumps(total_data)


def get_current_data(graph_type, alldata_string):
    times = []
    data = []
    if alldata_string is not None:
#        print("Getting current data: {} of length {}".format(alldata_string,len(alldata_string)))
        if len(alldata_string) > 0:
            alldata = json.loads(alldata_string)
            for id in alldata:
#                print(alldata[id])
#                print("")
                data.append(alldata[id][graph_type])
                times.append(alldata[id]['timestamp'])
        else:
            times.append([0])
            data.append([0])
    else:
        times.append([0])
        data.append([0])
    return times, data


@app.callback(Output('graphs','children'),
             [Input('graph-types', 'value')],
             [State('intermediate-values','children')])
def update_graphs(graph_names, current_data):
    graphs = []

    for graph_name in graph_names:
        times, graph_data = get_current_data(graph_name,current_data)
        traces = list()
        for i,itimes in enumerate(times):
            traces.append(go.Scatter(
            x=list(itimes),
            y=list(graph_data[i]),
            name='Scatter {}'.format(i),
            mode="lines"
            ))

        graphs.append(html.Div(dcc.Graph(
            id=graph_name,
            animate=True,
            figure={'data': traces,'layout' : go.Layout(xaxis=dict(range=[np.min(times),np.max(times)]),
                                                        yaxis=dict(range=[np.min(graph_data),np.max(graph_data)]),
                                                        margin={'l':50,'r':1,'t':45,'b':1},
                                                        title='{}'.format(graph_name))}
            )))

    return graphs


@app.callback(Output('raw_data','figure'),
             [Input('graph-update', 'n_intervals')],
             [State('intermediate-values','children'),
              State('start_button', 'n_clicks')])
def update_raw_data(n, current_data, n_clicks):
    if n_clicks is not None:
        times, graph_data = get_current_data('raw_data',current_data)
        traces = list()
        for i,itimes in enumerate(times):
            traces.append(go.Scatter(
            x=list(itimes),
            y=list(graph_data[i]),
            name='Scatter {}'.format(i),
            mode="lines"
            ))

        layout = go.Layout(xaxis=dict(range=[np.min(times),np.max(times)]),
                           yaxis=dict(range=[np.min(graph_data),np.max(graph_data)]),
                           margin={'l':50,'r':1,'t':45,'b':1},
                           title='{}'.format('raw_data'))

        return {'data': traces,'layout' : layout}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test", "-t",
        action='store_true',
        default=False,
    )

    args = parser.parse_args()
    arg_dict = vars(args)

    try:
        if not arg_dict['test']:
            clear_queue()
        app.run_server(debug=True)
    except:
        if not arg_dict['test']:
            send_queue_cmd('EXIT')
        # Still want to see traceback for debugging
        print('ERROR: GUI quit unexpectedly!')
        traceback.print_exc()
        pass



