import plotly.graph_objs as go
import plotly as py
import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html

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

from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton
from PyQt5.QtWidgets import QAction, QLineEdit, QMessageBox, QLabel
from PyQt5.QtWidgets import QMenu, QGridLayout, QFormLayout
from PyQt5.QtWidgets import QCheckBox, QFileDialog
from pyqtgraph import QtGui
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot, Qt, QUrl
from PyQt5.QtGui import QPalette, QFont, QTabWidget, QTabBar, QComboBox
from PyQt5.QtGui import QStyleFactory

TEXT_FONT = 16

dash_app = dash.Dash(__name__)

class App(QWidget):

    def __init__(self, nbins=4096, test=False, windows=False, **kwargs):
        super().__init__()
        self.title = 'DoseNet Sensor GUI'
        self.left = 0
        self.test_mode = test

        self.top = 80
        self.width = 1600
        self.height = 900

        self.start_time = None
        self.data_tabs = {}
        self.figures = {}
        self.data = {}
        self.fig_view = {}

        self.initUI()

    def initUI(self):
        self.setAutoFillBackground(True)
        self.setBackgroundRole(QPalette.Base)
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.initLayout()
        self.setLayout(self.layout)


    def initLayout(self):
        # Create Grid layout
        self.layout = QGridLayout()
        #self.layout.setSpacing(0.01)
        self.layout.setContentsMargins(0.,0.,0.,0.)

        # Create main plotting area
        self.tabs = QTabWidget(self)
        self.tabs.setStyleSheet("QTabWidget::tab-bar { alignment: left; } "+\
                "QTabWidget::pane { border: 2px solid #404040; } "+\
                "QTabBar {font-size: 14pt;}");
        tab_bar = QTabBar()
        tab_bar.setStyleSheet("QTabBar::tab { height: 40px; width: 250px;}")
        self.tabs.setTabBar(tab_bar)
        ptop, pleft, pheight, pwidth = 0, 0, 12, 12
        self.layout.addWidget(self.tabs,ptop,pleft,pheight,pwidth)
        self.setSelectionTab()

        textfont = QFont("Times", TEXT_FONT-2, QFont.Bold)
        label = QLabel('Select Plots', self)
        label.setFont(textfont)
        self.layout.addWidget(label,ptop,pleft+pwidth+1,1,1)
        label.setAlignment(Qt.AlignCenter)

        checkbox = QCheckBox("Diagnostic")
        checkbox.setFont(textfont)
        checkbox.setChecked(False)
        checkbox.stateChanged.connect(lambda:self.plotButtonState(checkbox))
        self.layout.addWidget(checkbox,ptop+1,pleft+pwidth+1,1,1,Qt.AlignHCenter)

        # Create push button
        self.addButton('Start',self.run,ptop+8,pleft+pwidth+1,1,1,"#66B2FF")
        self.addButton('Stop',self.stop,ptop+9,pleft+pwidth+1,1,1,"#FF6666")
        self.addButton('Clear',self.clear,ptop+10,pleft+pwidth+1,1,1,"#E0E0E0")


    def addButton(self,label,method,top,left,height,width,color="white"):
        '''
        Add a button to the main layout
        Inputs: label,
                method: button action function,
                location: top,left
                size: height,width
                color: background color for the button
        '''
        button = QPushButton(label, self)
        style_sheet_text = "background-color: "+color+";"+\
                           "border-style: outset;"+\
                           "border-width: 2px;"+\
                           "border-radius: 2px;"+\
                           "border-color: beige;"+\
                           "font: bold 25px;"+\
                           "min-width: 6em;"+\
                           "padding: 2px;"

        button.setStyleSheet(style_sheet_text)
        button.clicked.connect(method)
        self.layout.addWidget(button,top,left,height,width,Qt.AlignHCenter)


    def setSelectionTab(self):
        '''
        Fill out the tab providing all of the user input
            - user inputs include parameters for data acquisition
            - option to save data with user info for setting a unique file-name
        '''
        self.selection_tab = QWidget()
        self.tabs.addTab(self.selection_tab, "Configure")
        self.config_layout = QFormLayout()
        self.config_layout.setContentsMargins(60.,100.,60.,40.)

        ip_text = QLabel("IP Addresses:")
        textfont = QFont("Times", TEXT_FONT, QFont.Bold)
        ip_text.setFont(textfont)
        ip_text.setAlignment(Qt.AlignCenter)
        self.config_layout.addWidget(ip_text)
        self.ip_box = QLineEdit()
        self.ip_list = ""
        self.ip_box.textChanged.connect(self.updateIPList)
        self.config_layout.addWidget(self.ip_box)

        config_button = QPushButton("Load Configuration File", self)
        style_sheet_text = "background-color: #E0E0E0;"+\
                           "border-style: outset;"+\
                           "border-width: 2px;"+\
                           "border-radius: 2px;"+\
                           "border-color: beige;"+\
                           "font: bold 36px;"+\
                           "min-width: 6em;"+\
                           "padding: 2px;"

        config_button.setStyleSheet(style_sheet_text)
        config_button.clicked.connect(
                lambda:self.openConfigFiles())
        self.config_layout.addWidget(config_button)

        self.configfile_text = QLineEdit()
        self.configfile_text.textChanged.connect(self.updateFilename)
        self.config_layout.addWidget(self.configfile_text)

        self.selection_tab.setLayout(self.config_layout)

        checkbox = QCheckBox("Save Data")
        checkbox.setFont(QFont("Times", TEXT_FONT, QFont.Bold))
        checkbox.setChecked(False)
        checkbox.stateChanged.connect(lambda:self.setSaveData(checkbox))
        self.config_layout.addWidget(checkbox)


    def plotButtonState(self,b):
     if b.isChecked() == True:
        print("{} is selected".format(b.text()))
        self.setDataTab(b.text())
     else:
        print("{} is deselected".format(b.text()))
        self.rmvDataTab(b.text())


    def rmvSensorTab(self, tab_name):
        '''
        Remove Sensor Tab from GUI
        '''
        self.tabs.removeTab(self.data_tabs[tab_name][0])


    def setDataTab(self,tab_name):
        '''
        Setup the tab and layout for data plotting tab, initialize plots, etc.
        '''
        # Create canvas for plots
        if tab_name in self.data_tabs:
            self.tabs.insertTab(self.data_tabs[tab_name][0],
                                self.data_tabs[tab_name][1],tab_name)
            return

        self.initData(tab_name)
        itab = QWidget()
        index = self.tabs.addTab(itab, tab_name)
        self.data_tabs[tab_name] = [index,itab]
        tablayout = QGridLayout()
        tablayout.setSpacing(0.)
        tablayout.setContentsMargins(0.,0.,0.,0.)

        self.setPlots(tab_name,tablayout)
        itab.setLayout(tablayout)


    def initData(self,data_type):
        self.data[data_type] = [2, 1, 3, 1]


    def setPlots(self,data_type,layout):
        '''
        Set the initial plot layout, initialize plot curves, error bars, etc.
        '''
        if data_type == "Diagnostic":
            dash_app.layout = html.Div(
                [
                    dcc.Graph(id='live-graph', animate=True),
                    dcc.Interval(
                        id='graph-update',
                        interval=1*1000,
                        n_intervals=0
                    ),
                ]
            )
            @dash_app.callback(Output('live-graph', 'figure'),
                                         [Input('graph-update', 'n_intervals')])
            def update_diagnositic_graph(self,n):
                X = self.x_data
                Y = self.y_data

                data = go.Scatter(
                            x=list(X),
                            y=list(Y),
                            name='Scatter',
                            mode= 'lines+markers'
                        )

                return {'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]),
                                                            yaxis=dict(range=[min(Y),max(Y)]),)}

        print(data_type)
        fig = go.Figure(data=[{'type': 'scattergl', 'y': self.data[data_type]}])
        fig.update_layout(bargap=0,plot_bgcolor='rgba(0,0,0,0)')
        fig.update_yaxes(showgrid=True,gridcolor='black',linecolor='black',tickcolor='black',
                         tickfont=dict(color='black', size=16))
        fig.update_xaxes(linecolor='black',tickfont=dict(color='black', size=16))
        self.figures[data_type] = fig


        self.fig_view[data_type] = QWebEngineView()
        self.fig_view[data_type].load(QUrl("http://127.0.0.1:8050"))
        self.fig_view[data_type].show()
        #self.fig_view[data_type].raise_()
        #self.show_qt(data_type)
        layout.addWidget(self.fig_view[data_type],0,0,1,1)
        #layout.setRowStretch(1,1)


    def updatePlots(self):
        for data_type in self.data_tabs:
            self.updateData(data_type)
            self.updatePlot(data_type)        


    def updatePlot(self,data_type):
        self.figures[data_type].data[0].y = self.data[data_type]
        self.show_qt(data_type)


    def updateData(self,data_type):
        # Get new data from DAQ here
        #if self.test_mode:
        self.x_data.append(self.x_data[-1]+1)
        self.y_data.append(np.random.random(1)[0])


    def updateIPList(self):
        self.ip_list = self.ip_box.text()


    def openConfigFiles(self):
        #options = QFileDialog.Options()
        #options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Single File", "","All Files (*);")
        if fileName:
            print(fileName)
        self.configfile_text.setText(fileName)
        self.configfile_name = fileName


    def setSaveData(self,b):
        if b.isChecked() == True:
            print("Saving sensor data")
            self.saveData = True
        else:
            self.saveData = False


    def updateFilename(self,text):
        self.configfile_name = self.configfile_text.text()


    @pyqtSlot()
    def run(self):
        #Start DAQ here
        time_sample = 500
        if self.test_mode:
            time_sample = 1000
        print("Starting data collection")
        # Only set start time the first time user clicks start
        if self.start_time is None:
            self.start_time = float(format(float(time.time()), '.2f'))
        send_queue_cmd('START')

        #self.timer = QtCore.QTimer()
        #self.timer.timeout.connect(self.updatePlots)
        #self.timer.start(time_sample)

        dash_app.run_server(debug=True)


    @pyqtSlot()
    def stop(self):
        #Stop DAQ here
        if self.test_mode:
            print("Testing stop method")

    @pyqtSlot()
    def clear(self):
        #Clear plots here
        if self.test_mode:
            print("Testing clear method")


    def show_qt(self, data_type):
        raw_html = '<html><head><meta charset="utf-8" />'
        raw_html += '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script></head>'
        raw_html += '<body>'
        raw_html += po.plot(self.figures[data_type], include_plotlyjs=False, output_type='div')
        raw_html += '</body></html>'

        # setHtml has a 2MB size limit, need to switch to setUrl on tmp file
        # for large figures.
        self.fig_view[data_type].setHtml(raw_html)
        self.fig_view[data_type].show()
        self.fig_view[data_type].raise_()


    def exit(self):
        '''
        Send EXIT command to all sensors
        '''
        print("Exiting GUI")
        print("Sending EXIT command to all active sensors")
        send_queue_cmd('EXIT')
        time.sleep(2)

#-------------------------------------------------------------------------------
# Methods for communication with the shared queue
#   - allows commmunication between GUI and sensor DAQs
#   - send commands and receive sensor data
#-------------------------------------------------------------------------------
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test", "-t",
        action='store_true',
        default=False,
    )

    args = parser.parse_args()
    arg_dict = vars(args)

    global ex
    # Wrap everything in try/except so that sensor DAQs can be shutdown cleanly
    try:
        if not arg_dict['test']:
            clear_queue()
        app = QApplication(sys.argv)
        QApplication.setStyle(QStyleFactory.create("Cleanlooks"))

        ex = App(**arg_dict)
        ex.show()

        atexit.register(ex.exit)
    except:
        if not arg_dict['test']:
            send_queue_cmd('EXIT')
        # Still want to see traceback for debugging
        print('ERROR: GUI quit unexpectedly!')
        traceback.print_exc()
        pass

    ret = app.exec_()
    print(ret)
    sys.exit(ret)

