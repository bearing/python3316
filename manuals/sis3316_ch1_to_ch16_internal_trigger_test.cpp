/***************************************************************************/

/* Modifications/notes by Matt Proveaux are denoted with a -m@             */


// reconfigured for multiple 3316s  -jfr 2014mar12


// sis3316_ch1_to_ch16_internal_trigger_test.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.
//
/***************************************************************************/
/*                                                                         */
/*  Filename: sis3316_ch1_to_ch16_internal_trigger_test.cpp                */
/*                                                                         */
/*  Funktion:                                                              */
/*                                                                         */
/*  Autor:                TH                                               */
/*  date:                 28.01.2013                                       */
/*  last modification:    28.01.2013                                       */
/*                                                                         */
/* ----------------------------------------------------------------------- */
/*                                                                         */
/*  SIS  Struck Innovative Systeme GmbH                                    */
/*                                                                         */
/*  Harksheider Str. 102A                                                  */
/*  22399 Hamburg                                                          */
/*                                                                         */
/*  Tel. +49 (0)40 60 87 305 0                                             */
/*  Fax  +49 (0)40 60 87 305 20                                            */
/*                                                                         */
/*  http://www.struck.de                                                   */
/*                                                                         */
/*  © 2013                                                                 */
/*                                                                         */
/*                                                                         */
/***************************************************************************/

//define LINUX    // else WINDOWS


#ifndef LINUX
	#define WINDOWS
#endif






#define CERN_ROOT_PLOT

#ifdef CERN_ROOT_PLOT
   #include "TApplication.h"
   #include "TObject.h"
   #include "TH1.h"
   #include "TH1D.h"
   #include "TH1F.h"
   #include "TH2D.h"
   #include "TGraph.h"
   #include "TMultiGraph.h"
   #include "TMath.h"
   #include "TCanvas.h"
    //#include "TRandom.h"
   //#include "TThread.h"
   #include <TSystem.h>
   #include "TLatex.h"
   #include "TGNumberEntry.h"
   #include "TRootEmbeddedCanvas.h"

   #pragma comment (lib, "libRio")
   #pragma comment (lib, "libcore")
   #pragma comment (lib, "libHist")
   #pragma comment (lib, "libTree")
   #pragma comment (lib, "libgpad")
   #pragma comment (lib, "libcint")
   #pragma comment (lib, "libGraf")
   #pragma comment (lib, "libGui")

#endif


//#include "stdafx.h"

#include <Windows.h>
#include <tchar.h>
#include <stdio.h>
#include <iostream>

#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <fcntl.h>
#include <time.h>
#include <math.h>
#include <cstdio>
#include <ctime>

using namespace std;

/******************************************************************************************************/

// choose VME Interface
//#define PCI_VME_INTERFACE		// sis1100/3100 optical interface
#define USB_VME_INTERFACE


#ifdef PCI_VME_INTERFACE
	#include "sis1100w_vme_class.h"
#endif

#ifdef USB_VME_INTERFACE
	#include "sis3150w_vme_class.h"
#endif

#include "sis3316_class.h"
//sis3316_adc *gl_sis3316_adc1 ;

#define MAX_NOF_SIS3316_ADCS			4 //jfr 2014mar10 changed from 1 to 4
//#define BROADCAST_BASE_ADDR				0x40000000 
#define FIRST_MODULE_BASE_ADDR			0x40000000  // jfr 2014mar 11 changed from 0x31000000
#define MODULE_BASE_OFFSET				0x10000000  // jfr 2014mar11 changed from 0x01000000 

#include "sis3316.h"

/******************************************************************************************************/

#ifdef CERN_ROOT_PLOT

#include "sis3316_cern_root_class.h"  
sis_root_graph *gl_graph_raw ;

sis_root_channel_energy_histogram *gl_channel_energy_histogram ;

sis_root_graph_maw *gl_graph_maw ;
sis_root_intensity_graph *gl_intensity_raw ;

unsigned int gl_graph_zoom_raw_draw_length ;
unsigned int gl_graph_zoom_raw_draw_bin_offset ;

#endif
/******************************************************************************************************/


/*===========================================================================*/
/* Globals					  			     */
/*===========================================================================*/
#define MAX_NUMBER_LWORDS_64MBYTE			0x1000000       /* 64MByte */ 
//i.e. each bank has 64 MByte memory, see manual p.49 
// in a 32 bit system: 1 word = 4 bytes. 
// 0x1000000 = 16777216 words -> 16777216 words* 4 bytes/word* 1 Mbyte/(1024^2 bytes) *  = 64 MBytes -m@ 


//#define MAX_NUMBER_LWORDS_64MBYTE			0x1000000       /* 64MByte */

unsigned int gl_wblt_data[MAX_NUMBER_LWORDS_64MBYTE] ;
unsigned int gl_rblt_data[MAX_NUMBER_LWORDS_64MBYTE] ;

BOOL gl_stopReq = FALSE;


FILE *gl_FILE_DataEvenFilePointer           ;

/*===========================================================================*/
/* Prototypes			                               		  			     */
/*===========================================================================*/

int WriteBufferHeaderCounterNofChannelToDataFile (unsigned int buffer_no,unsigned int nof_events, unsigned int event_length);
int WriteEventsToDataFile (unsigned int* memory_data_array, unsigned int nof_write_length_lwords);

int WriteDataFileHeader(unsigned int headerlength, unsigned int samplelength, unsigned int mawtestbufferlength); 
//Write Data file header -m@
int WriteEOF(); //Write EOF -m@ 

void program_stop_and_wait(void);
BOOL CtrlHandler( DWORD ctrlType );
void usleep(unsigned int uint_usec) ;


int _tmain(int argc, _TCHAR* argv[])
{

CHAR char_messages[128];
UINT nof_found_devices ;

int return_code ;
unsigned int first_mod_base, nof_modules   ;
unsigned int module_base_addr_array[MAX_NOF_SIS3316_ADCS]   ; // jfr
unsigned int i, i_mod, module_index;                          // jfr

unsigned int module_base_addr  ;
unsigned int addr, data;
unsigned int loop_counter;
unsigned int error_loop_counter;

unsigned int fp_lvds_bus_ctrl_value[MAX_NOF_SIS3316_ADCS]  ;  // jfr

UINT req_nof_32bit_words, got_nof_32bit_words;
unsigned int sample_length;
unsigned int sample_start_index;
unsigned int trigger_gate_window_length;
unsigned int address_threshold;
unsigned int nof_events;
unsigned int pre_trigger_delay ;
unsigned int bank1_armed_flag ;
unsigned int poll_counter ;
unsigned int i_adc;
unsigned int sis3316_not_OK;
unsigned int memory_bank_offset_addr ;
//unsigned int ch_last_bank_address_array[16];
//unsigned int ch_event_counter_array[16];
unsigned int ch_event_counter;

unsigned int event_length;
unsigned int header_length;
unsigned int maw_length;

unsigned int iob_delay_value ;
unsigned int clock_source_choice ;
unsigned int p_val ;
unsigned int g_val ;
unsigned int trigger_threshold_value ;


unsigned int i_adc_fpga ;
unsigned int i_ch ;


unsigned int nof_lwords ;
unsigned int written_nof_words ;

unsigned int timeout_counter ;
unsigned int peak_test_error_counter ;
unsigned int gate_test_error_counter ;
unsigned int maw_value_error_counter ;

unsigned short* ushort_adc_buffer_ptr;

unsigned int maw_test_buffer_length ;
unsigned int maw_test_buffer_delay ;

unsigned int header_accu_6_values_enable_flag ;
unsigned int header_accu_2_values_enable_flag ;
unsigned int header_maw_3_values_enable_flag ;
unsigned int maw_test_buffer_enable_flag ;

unsigned int header_maw_3_values_offset ;
unsigned int header_accu_6_values_offset ;
unsigned int header_accu_2_values_offset ;

unsigned int uint_pileup ;      //jfr
unsigned int uint_re_pileup ;   //jfr

unsigned int i_gate;
unsigned int gate_length[8];
unsigned int gate_start_index[8];

unsigned int online_accumulator[8] ;
unsigned int offline_accumulator[8] ;

unsigned online_peak_index, online_peak_value ;
unsigned offline_peak_index, offline_peak_value ;

signed offline_maw_max ;
signed offline_maw_value ;

signed read_maw_max ;
signed read_maw_befor_trigger ;
signed read_maw_with_trigger ;
float float_diff_N_to_MaxDiv2; ;
float float_diff_Nm1_to_N; ;
float float_timestamp_correctur ;

			signed int offline_maw_max_50;
			unsigned int maw_befor_with_trigger_ok_flag ;
			signed offline_maw_befor_trigger ;
			signed offline_maw_with_trigger ;



unsigned int uint_DataEvent_OpenFlag ;
   


/******************************************************************************************************************************/
/* VME Master Create, Open and Setup                                                                                          */
/******************************************************************************************************************************/
 

#ifdef PCI_VME_INTERFACE
	// create SIS1100/SIS310x vme interface device
	sis1100 *vme_crate = new sis1100(0);
#endif

#ifdef USB_VME_INTERFACE
USHORT idVendor;
USHORT idProduct;
USHORT idSerNo;
USHORT idFirmwareVersion;
USHORT idDriverVersion;
	// create SIS3150USB vme interface device
	sis3150 *vme_crate = new sis3150(0);
#endif

// open Vme Interface device
	return_code = vme_crate->vmeopen ();  // open Vme interface
	vme_crate->get_vmeopen_messages (char_messages, &nof_found_devices);  // open Vme interface
	printf("\n%s    (found %d vme interface device[s])\n\n",char_messages, nof_found_devices);

	if(return_code != 0x0) {
		//printf("ERROR: vme_crate->vmeopen: return_code = 0x%08x\n\n", return_code);
		program_stop_and_wait();
		return -1 ;
	}


/******************************************************************************************/
// additional Vme interface device informations
#ifdef USB_VME_INTERFACE
	vme_crate->get_device_informations (&idVendor, &idProduct, &idSerNo, &idFirmwareVersion, &idDriverVersion);  //  
	printf("idVendor:           %04X\n",idVendor);
	printf("idProduct:          %04X\n",idProduct);
	printf("idSerNo:            %d\n",idSerNo);
	printf("idFirmwareVersion:  %04X\n",idFirmwareVersion);
	printf("idDriverVersion:    %04X\n",idDriverVersion);
	printf("\n\n");

#endif
/******************************************************************************************/
	if( !SetConsoleCtrlHandler( (PHANDLER_ROUTINE)CtrlHandler, TRUE ) ){
		printf( "Error setting Console-Ctrl Handler\n" );
		return -1;
	}
/******************************************************************************************/

/******************************************************************************************/
	//for (i_mod=0; i_mod<MAX_NOF_SIS3316_ADCS; i_mod++) {
	//	module_base_addr_array[i_mod] = FIRST_MODULE_BASE_ADDR + (i_mod * MODULE_BASE_OFFSET);
	//}

	//hardcoding addresses
	module_base_addr_array[0] = 0x50000000;
	module_base_addr_array[1] = 0x60000000;
	module_base_addr_array[2] = 0x70000000;
	module_base_addr_array[3] = 0x80000000;

	//module_base_addr = 0x60000000 ; jfr commented out 2014mar11
	for (i_mod=0; i_mod<MAX_NOF_SIS3316_ADCS; i_mod++) {
		module_base_addr = module_base_addr_array[i_mod] ;
		return_code = vme_crate->vme_A32D32_read ( module_base_addr + SIS3316_MODID, &data);  
		//printf("vme_A32D32_read: data = 0x%08x     return_code = 0x%08x\n", data, return_code);
		printf("SSIS3316 #%d  with addr = 0x%08\n\n", i_mod+1, module_base_addr);
		sis3316_not_OK = 0 ;
		if (return_code == 0) {
			printf("SIS3316_MODID                    = 0x%08x\n\n", data);
			vme_crate->vme_A32D32_read ( module_base_addr + SIS3316_SERIAL_NUMBER_REG, &data);  
			printf("SIS3316_SERIAL_NUMBER_REG        = %d\n\n", data);

			vme_crate->vme_A32D32_write ( module_base_addr + SIS3316_ADC_CH1_4_INPUT_TAP_DELAY_REG, 0x400 ); // Clear Link Error Latch bits
			vme_crate->vme_A32D32_write ( module_base_addr + SIS3316_ADC_CH5_8_INPUT_TAP_DELAY_REG, 0x400 ); // Clear Link Error Latch bits
			vme_crate->vme_A32D32_write ( module_base_addr + SIS3316_ADC_CH9_12_INPUT_TAP_DELAY_REG, 0x400 ); // Clear Link Error Latch bits
			vme_crate->vme_A32D32_write ( module_base_addr + SIS3316_ADC_CH13_16_INPUT_TAP_DELAY_REG, 0x400 ); // Clear Link Error Latch bits

			vme_crate->vme_A32D32_read ( module_base_addr + SIS3316_ADC_CH1_4_FIRMWARE_REG, &data);  
			printf("SIS3316_ADC_CH1_4_FIRMWARE_REG   = 0x%08x \n", data);
			vme_crate->vme_A32D32_read ( module_base_addr + SIS3316_ADC_CH5_8_FIRMWARE_REG, &data);  
			printf("SIS3316_ADC_CH5_8_FIRMWARE_REG   = 0x%08x \n", data);
			vme_crate->vme_A32D32_read ( module_base_addr + SIS3316_ADC_CH9_12_FIRMWARE_REG, &data);  
			printf("SIS3316_ADC_CH9_12_FIRMWARE_REG  = 0x%08x \n", data);
			vme_crate->vme_A32D32_read ( module_base_addr + SIS3316_ADC_CH13_16_FIRMWARE_REG, &data);  
			printf("SIS3316_ADC_CH13_16_FIRMWARE_REG = 0x%08x \n\n", data);

			vme_crate->vme_A32D32_write ( module_base_addr + SIS3316_VME_FPGA_LINK_ADC_PROT_STATUS, 0xE0E0E0E0);  // clear error Latch bits 
			vme_crate->vme_A32D32_read ( module_base_addr + SIS3316_VME_FPGA_LINK_ADC_PROT_STATUS, &data);  
			printf("SIS3316_VME_FPGA_LINK_ADC_PROT_STATUS: data = 0x%08x     return_code = 0x%08x\n", data, return_code);
			if (data != 0x18181818) { sis3316_not_OK = 1 ; }

			vme_crate->vme_A32D32_read ( module_base_addr + SIS3316_ADC_CH1_4_STATUS_REG, &data);  
			printf("SIS3316_ADC_CH1_4_STATUS_REG     = 0x%08x \n", data);
			if (data != 0x130118) { sis3316_not_OK = 1 ; } // 0x130018 is for sis3316 modules with serial # 1-10 (slower data link speed)

			vme_crate->vme_A32D32_read ( module_base_addr + SIS3316_ADC_CH5_8_STATUS_REG, &data);  
			printf("SIS3316_ADC_CH5_8_STATUS_REG     = 0x%08x \n", data);
			if (data != 0x130118) { sis3316_not_OK = 1 ; }

			vme_crate->vme_A32D32_read ( module_base_addr + SIS3316_ADC_CH9_12_STATUS_REG, &data);  
			printf("SIS3316_ADC_CH9_12_STATUS_REG    = 0x%08x \n", data);
			if (data != 0x130118) { sis3316_not_OK = 1 ; }

			vme_crate->vme_A32D32_read ( module_base_addr + SIS3316_ADC_CH13_16_STATUS_REG, &data);  
			printf("SIS3316_ADC_CH13_16_STATUS_REG   = 0x%08x \n\n", data);
			if (data != 0x130118) { sis3316_not_OK = 1 ; }

		}
		else {
			printf("SIS3316_MODID                  = 0x%08x     return_code = 0x%08x\n", data, return_code);
			program_stop_and_wait() ;
		}

		if (sis3316_not_OK != 0) {
		printf("sis3316_not_OK                 \n");
		program_stop_and_wait() ;
		}
		printf("\n");
	}

	sis3316_adc  *sis3316_adc_array[MAX_NOF_SIS3316_ADCS] ;
	for (i_mod=0; i_mod<MAX_NOF_SIS3316_ADCS; i_mod++) {
		sis3316_adc_array[i_mod] = new sis3316_adc( vme_crate, module_base_addr_array[i_mod]);
	}


/******************************************************************************************************************************/
/* CERN ROOT                                                                                                                  */
/******************************************************************************************************************************/

#ifdef CERN_ROOT_PLOT

	int root_graph_x ;
	int root_graph_y ;
	int root_graph_x_size ;
	int root_graph_y_size ;
	char root_graph_text[80] ;
	//char root_graph_text_hist[80] ; //jfr 08122015


	root_graph_x_size = 700 ;
	root_graph_y_size = 300 ;      // this adjusts the actual size of the window, not the scale. 

	root_graph_x = 500 ; 	      
	root_graph_y = 350 ; 	      //this adjusts the position of the window on the screen 



	TApplication theApp("SIS3316 Application: Test", &argc, (char**)argv);
	//sis_root_graph *graph_raw = new sis_root_graph(root_graph_text, root_graph_x, root_graph_y, root_graph_x_size, root_graph_y_size) ;
	strcpy(root_graph_text,"SIS3316 Graph: Raw data") ;
	gl_graph_raw      = new sis_root_graph(root_graph_text, root_graph_x, root_graph_y, root_graph_x_size, root_graph_y_size) ;
	
	//strcpy(root_graph_text_hist,"SIS3316 Graph: Energy Histogram") ; //jfr 08122015
	//gl_channel_energy_histogram = new sis_root_channel_energy_histogram(root_graph_text_hist, 0, 0, root_graph_x_size, root_graph_y_size) ; //jfr 08122015

#endif



	
/******************************************************************************************/

/******************************************************************************************/
	for (i_mod=0; i_mod<MAX_NOF_SIS3316_ADCS; i_mod++)  {
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_KEY_RESET, 0);  
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_KEY_DISARM , 0);  //   
	}


	i_mod = 0;
	iob_delay_value = 0x48 ; // = 72 decimal -m@ 
	clock_source_choice = 0 ; // 250MHz
	//clock_source_choice = 1 ; // 125MHz
	//clock_source_choice = 2 ; // 62.5MHz

	// set clock , wait 20ms in sis3316_adc1->set_frequency
	// reset DCM in sis3316_adc1->set_frequency
	
	switch (clock_source_choice) {
	    case 0:
			sis3316_adc_array[i_mod]->set_frequency(0, sis3316_adc_array[i_mod]->freqPreset250MHz);
			iob_delay_value = 0x48 ;
			break;
	    case 1:
			sis3316_adc_array[i_mod]->set_frequency(0, sis3316_adc_array[i_mod]->freqPreset125MHz);
			iob_delay_value = 0x48 ;
			break;
	    case 2:
			sis3316_adc_array[i_mod]->set_frequency(0, sis3316_adc_array[i_mod]->freqPreset62_5MHz);
			iob_delay_value = 0x0 ;
			break;
	}
	

	// Setup of Sample Clock on Frontpanel LVDS Bus:
	// first SIS3316 drives the Clock
	data = 0 ;
	data = data + 0x1 ;   // Enables the Control lines to the FP-Bus  
	data = data + 0x2 ;   // Enables the Status lines to the FP-Bus
	data = data + 0x10 ;  // Enables Sample Clock driver to the FP-Bus
	data = data + 0x0 ;   // Selects internal Clock oscillator
	//data = data + 0x20 ;   // Selects Lemo Clock In 
	data = data + 0x4 ;   // temp
	i_mod=0;	
	fp_lvds_bus_ctrl_value[i_mod] = data ; // first module

	data = 0x2 ;   // // Enables the Status lines to the FP-Bus, only 
	data = data + 0x4 ;   // temp
	for (i_mod=1; i_mod<MAX_NOF_SIS3316_ADCS; i_mod++) {
		fp_lvds_bus_ctrl_value[i_mod] = data ; // // next modules
	}

	for (i_mod=0; i_mod<MAX_NOF_SIS3316_ADCS; i_mod++) {
		vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_FP_LVDS_BUS_CONTROL, fp_lvds_bus_ctrl_value[i_mod] ); //  
	}

	// define the sample Clock on each module	
	//data = 0 ; // Onboard Oscillator
	//data = 1 ; // VXS-Bus Clock
	data = 2 ; // FP-LVDS-Bus Clock  
	//data = 3 ; // External NIM Clock
	for (i_mod=0; i_mod<MAX_NOF_SIS3316_ADCS; i_mod++) {
		return_code = vme_crate->vme_A32D32_write (  module_base_addr_array[i_mod] + SIS3316_SAMPLE_CLOCK_DISTRIBUTION_CONTROL, data ); //  
	}

	// Sample Clock DCM/PLL Reset on each SIS3316 FPGA ; used for internal logic
	for (i_mod=0; i_mod<MAX_NOF_SIS3316_ADCS; i_mod++) {
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_KEY_ADC_CLOCK_DCM_RESET, 0x0 ); //  
	}
	// min. 10ms wait or check if PLL is locked
	Sleep(20);



	// set IOB _delay  
	for (i_mod=0; i_mod<MAX_NOF_SIS3316_ADCS; i_mod++) {
		Sleep(1) ;
		return_code = vme_crate->vme_A32D32_write (  module_base_addr_array[i_mod] + SIS3316_ADC_CH1_4_INPUT_TAP_DELAY_REG, 0xf00 ); // Calibrate IOB _delay Logic
		return_code = vme_crate->vme_A32D32_write (  module_base_addr_array[i_mod] + SIS3316_ADC_CH5_8_INPUT_TAP_DELAY_REG, 0xf00 ); // Calibrate IOB _delay Logic
		return_code = vme_crate->vme_A32D32_write (  module_base_addr_array[i_mod] + SIS3316_ADC_CH9_12_INPUT_TAP_DELAY_REG, 0xf00 ); // Calibrate IOB _delay Logic
		return_code = vme_crate->vme_A32D32_write (  module_base_addr_array[i_mod] + SIS3316_ADC_CH13_16_INPUT_TAP_DELAY_REG, 0xf00 ); // Calibrate IOB _delay Logic
		Sleep(1) ;
		return_code = vme_crate->vme_A32D32_write (  module_base_addr_array[i_mod] + SIS3316_ADC_CH1_4_INPUT_TAP_DELAY_REG, 0x300 + iob_delay_value ); // set IOB _delay Logic
		return_code = vme_crate->vme_A32D32_write (  module_base_addr_array[i_mod] + SIS3316_ADC_CH5_8_INPUT_TAP_DELAY_REG, 0x300 + iob_delay_value ); // set IOB _delay Logic
		return_code = vme_crate->vme_A32D32_write (  module_base_addr_array[i_mod] + SIS3316_ADC_CH9_12_INPUT_TAP_DELAY_REG, 0x300 + iob_delay_value ); // set IOB _delay Logic
		return_code = vme_crate->vme_A32D32_write (  module_base_addr_array[i_mod] + SIS3316_ADC_CH13_16_INPUT_TAP_DELAY_REG, 0x300 + iob_delay_value ); // set IOB _delay Logic
		Sleep(1) ;
	}
	
	for (i_mod=0; i_mod<MAX_NOF_SIS3316_ADCS; i_mod++) {
		for (i_adc_fpga=0;i_adc_fpga<4;i_adc_fpga++) { // over all 4 ADC-FPGAs
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH1_4_SPI_CTRL_REG, 0x01000000 ); // enable ADC outputs (bit was cleared with Key-reset)
		}
	}

// Gain/Termination  
	for (i_mod=0; i_mod<MAX_NOF_SIS3316_ADCS; i_mod++) {
		data = 0 ;
		//data = data + 0x00040404 ; // disable 50 Ohm Termination on ch1/ch2/ch3
		//data = data + 0x04040404 ; // disable 50 Ohm Termination
		//data = data + 0x01010101 ; // 2 V Range -m@ 
		data = data + 0x00000000; // 5 V Range -m@   ---- See p. 90 
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH1_4_ANALOG_CTRL_REG, data); //  
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH5_8_ANALOG_CTRL_REG, data ); //  
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH9_12_ANALOG_CTRL_REG, data ); // 
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH13_16_ANALOG_CTRL_REG, data ); //  
	}

	//channel Header ID
	for (i_mod=0; i_mod<MAX_NOF_SIS3316_ADCS; i_mod++) {
		data =  module_base_addr_array[i_mod] + 0x0 ;
		return_code = vme_crate->vme_A32D32_write (  module_base_addr_array[i_mod] + SIS3316_ADC_CH1_4_CHANNEL_HEADER_REG, data); //  
		data =  module_base_addr_array[i_mod] + 0x00400000 ;
		return_code = vme_crate->vme_A32D32_write (  module_base_addr_array[i_mod] + SIS3316_ADC_CH5_8_CHANNEL_HEADER_REG, data ); //  
		data =  module_base_addr_array[i_mod] + 0x00800000 ;
		return_code = vme_crate->vme_A32D32_write (  module_base_addr_array[i_mod] + SIS3316_ADC_CH9_12_CHANNEL_HEADER_REG, data ); // 
		data =  module_base_addr_array[i_mod] + 0x00C00000 ;
		return_code = vme_crate->vme_A32D32_write (  module_base_addr_array[i_mod] + SIS3316_ADC_CH13_16_CHANNEL_HEADER_REG, data ); //  
	}

	
	nof_events = 1 ;
	trigger_gate_window_length = 100;	
	//cout << "Sample Length = "; //-m@
	//cin >> sample_length; 
	sample_length              = 70;	//this is the number of samples taken for a triggering signal -m@ 
	sample_start_index         = 0;	
	maw_test_buffer_length = 100 ;   



	p_val = 10 ;  
	g_val = 10 ; //these values are respectively the Peaking time and Gap time as defined in the 3316 manual on p. 36 FIR filter for trigger generation -m@ 

	//pre_trigger_delay = 50 ;
	pre_trigger_delay = p_val + (p_val>>1) + g_val + 16 + 11;  // 16 is additional delay for 50% CFD trigger --> see edge at index 10 //+11 jfr
	//this is the time delay, see p. 35 "Delayed MA". -m@

	// pre_trigger_delay increases or decreases the length (i.e. number of samples) BEFORE the triggering signal, which are then written to file or displayed on the 

	header_accu_6_values_enable_flag = 1 ; //initially enabled -m@ //
	header_accu_2_values_enable_flag = 0 ;
	header_maw_3_values_enable_flag  = 0 ;//initially enabled -m@ //enabled 07oct13 jfr

	maw_test_buffer_enable_flag = 0; // enables the output of MAW data to buffer -m@

	trigger_threshold_value = 1500*p_val; //  for 5V scale this is equivalent to 0.147 V, i.e. (480/2^14)*5V // changed from 480 to 900 28mar2014 jfr
	// we multiply by p_value because we actually trigger on the MAW which has a higher amplitude due to signal summing. 

	uint_pileup = 75;  //  
	uint_re_pileup = 62;  //   


	// internal Trigger Setup
	for (i_mod=0; i_mod<MAX_NOF_SIS3316_ADCS; i_mod++) {
		// disable all FIR Triggers
		// it is enabled later on -m@
		data = 0x00000000 ;
		for (i_adc_fpga=0; i_adc_fpga<4; i_adc_fpga++) {
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH1_4_SUM_FIR_TRIGGER_THRESHOLD_REG, data );  // disable all ch_sum
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH1_FIR_TRIGGER_THRESHOLD_REG, data );  // disable ch1, 5, 9, 13 
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH2_FIR_TRIGGER_THRESHOLD_REG, data );  // disable ch2, ..
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH3_FIR_TRIGGER_THRESHOLD_REG, data );  // disable ch3, ..
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH4_FIR_TRIGGER_THRESHOLD_REG, data );  // disable ch4, ..
		}	


	// set HighEnergy Threshold
	data =  0x08000000 + (p_val * 50000) ; // gt 1000   //changed from 1000 to 50000 -jfr 29aug2015
		for (i_adc_fpga=0; i_adc_fpga<4; i_adc_fpga++) {
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH1_FIR_HIGH_ENERGY_THRESHOLD_REG, data);  
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH2_FIR_HIGH_ENERGY_THRESHOLD_REG, data);  
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH3_FIR_HIGH_ENERGY_THRESHOLD_REG, data);  
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH4_FIR_HIGH_ENERGY_THRESHOLD_REG, data);  
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH1_4_SUM_FIR_HIGH_ENERGY_THRESHOLD_REG, data);  
		}	


	// set FIR Trigger Setup
	for (i_adc_fpga=0;i_adc_fpga<4;i_adc_fpga++) {
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH1_4_SUM_FIR_TRIGGER_SETUP_REG, 0) ; // clear FIR Trigger Setup -> a following Setup will reset the logic ! 
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH1_4_SUM_FIR_TRIGGER_SETUP_REG, p_val + (g_val << 12)) ;  
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH1_FIR_TRIGGER_SETUP_REG, 0) ;  
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH1_FIR_TRIGGER_SETUP_REG, p_val + (g_val << 12)) ;  
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH2_FIR_TRIGGER_SETUP_REG, 0) ;  
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH2_FIR_TRIGGER_SETUP_REG, p_val + (g_val << 12)) ;  
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH3_FIR_TRIGGER_SETUP_REG, 0) ;  
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH3_FIR_TRIGGER_SETUP_REG, p_val + (g_val << 12)) ;  
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH4_FIR_TRIGGER_SETUP_REG, 0) ;  
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH4_FIR_TRIGGER_SETUP_REG, p_val + (g_val << 12)) ;  
		}

// configure FIR  trigger logic	

	//for loop sets a given trigger value for each of the 16 registers.  -m@ 
	data =  0xB0000000 + 0x08000000 + trigger_threshold_value ;  // cfd 50 // //this gives a integer value, which can be converted to bits--these correspond to the 32 assignment bits for different functions... -m@ 
	for (i_adc_fpga=0;i_adc_fpga<4;i_adc_fpga++) {
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH1_4_SUM_FIR_TRIGGER_THRESHOLD_REG, data ); // -tja 2013/sep/18 
		//return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH1_FIR_TRIGGER_THRESHOLD_REG, 0x80000000 + 0x08000000 + trigger_threshold_value);  // no cfd 50  
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH1_FIR_TRIGGER_THRESHOLD_REG, data);  //  
		//return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH1_FIR_TRIGGER_THRESHOLD_REG, 0xA0000000 + 0x08000000 + trigger_threshold_value);  // zero crossing  
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH2_FIR_TRIGGER_THRESHOLD_REG, data);  //   
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH3_FIR_TRIGGER_THRESHOLD_REG, data);  //  
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH4_FIR_TRIGGER_THRESHOLD_REG, data);  //  
	}

	//why are these next two lines here?  jfr 2014mar12
	data =  0xB0000000 + 0x08000000 + trigger_threshold_value ;  
	return_code=vme_crate->vme_A32D32_write(module_base_addr_array[i_mod] + SIS3316_ADC_CH5_FIR_TRIGGER_THRESHOLD_REG, data); 
	
	}
	
	for (i_mod=0; i_mod<MAX_NOF_SIS3316_ADCS; i_mod++) {
		return_code = vme_crate->vme_A32D32_write (  module_base_addr_array[i_mod] + SIS3316_VME_FPGA_LINK_ADC_PROT_STATUS, 0xE0E0E0E0);  // clear error Latch bits 
	}

#ifdef raus // is already done in "new sis3316_adc( vme_crate, module_base_addr_array[i_mod]);"
	// set ADC chips via SPI
	for (i_mod=0; i_mod<MAX_NOF_SIS3316_ADCS; i_mod++) {
		for (i_adc_fpga=0;i_adc_fpga<4;i_adc_fpga++) {I
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH1_4_SPI_CTRL_REG, 0x81001444 ); // SPI (OE)  set binary
			usleep(1); //unsigned int uint_usec  
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH1_4_SPI_CTRL_REG, 0x81401444 ); // SPI (OE)  set binary
			usleep(1); //unsigned int uint_usec  
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH1_4_SPI_CTRL_REG, 0x8100ff01 ); // SPI (OE)  update
			usleep(1); //unsigned int uint_usec  
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH1_4_SPI_CTRL_REG, 0x8140ff01 ); // SPI (OE)  update
			usleep(1); //unsigned int uint_usec  
		}	
	}
#endif

//  set ADC offsets (DAC)     ///This configuration is for an adc-input range from 0 to -5V   
	for (i_mod=0; i_mod<MAX_NOF_SIS3316_ADCS; i_mod++) {
		data = 0x8000;  // middle
		for (i_adc_fpga=0;i_adc_fpga<4;i_adc_fpga++) { // over all 4 ADC-FPGAs
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH1_4_DAC_OFFSET_CTRL_REG, 0x80000000 + 0x8000000 +  0xf00000 + 0x1);  // set internal Reference
			usleep(1); //unsigned int uint_usec  
			//return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH1_4_DAC_OFFSET_CTRL_REG, 0x80000000 + 0x2000000 +  0xf00000 + ((0xd000 & 0xffff) << 4) );  // This was what originally used -m@
			//return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH1_4_DAC_OFFSET_CTRL_REG, 0x80000000 + 0x2000000 +  0xf00000 + ((0x8000 & 0xffff) << 4) );  
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH1_4_DAC_OFFSET_CTRL_REG, 0x80000000 + 0x2000000 +  0xf00000 + ((0 & 0xffff) << 4) ); /// 0 to -5 V offset. - m@ 
		
			// this was origionally commented out, I'm using this now as it sets up the correct offset for 0 to -5 V . // Adjust the number before the "&" will shift the offset corresponding to the values on p.92 -m@ 
		
			// note that the last hex number correponds to "DAC Data Bits 0-15" in ADC offset (DAC) control registers. The bit operation <<, "shift left" accounts for bits 0-3 at these registers, which are unused. In any case, the DAC Data Bits control  the DAC offset, see manual p. 92 -m@  
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + (i_adc_fpga*SIS3316_FPGA_ADC_REG_OFFSET) + SIS3316_ADC_CH1_4_DAC_OFFSET_CTRL_REG, 0xC0000000 );  // 
			usleep(1); //unsigned int uint_usec  
		}
	}



	for (i_mod=0; i_mod<MAX_NOF_SIS3316_ADCS; i_mod++) {
		return_code = vme_crate->vme_A32D32_write (  module_base_addr_array[i_mod] + SIS3316_ADC_CH1_4_TRIGGER_GATE_WINDOW_LENGTH_REG, (trigger_gate_window_length -2 & 0xffff) ); // trigger_gate_window_length
		return_code = vme_crate->vme_A32D32_write (  module_base_addr_array[i_mod] + SIS3316_ADC_CH5_8_TRIGGER_GATE_WINDOW_LENGTH_REG, (trigger_gate_window_length -2 & 0xffff) ); // trigger_gate_window_length  
		return_code = vme_crate->vme_A32D32_write (  module_base_addr_array[i_mod] + SIS3316_ADC_CH9_12_TRIGGER_GATE_WINDOW_LENGTH_REG, (trigger_gate_window_length -2 & 0xffff) ); // trigger_gate_window_length   
		return_code = vme_crate->vme_A32D32_write (  module_base_addr_array[i_mod] + SIS3316_ADC_CH13_16_TRIGGER_GATE_WINDOW_LENGTH_REG, (trigger_gate_window_length -2 & 0xffff) ); // trigger_gate_window_length

		return_code = vme_crate->vme_A32D32_write (  module_base_addr_array[i_mod] + SIS3316_ADC_CH1_4_RAW_DATA_BUFFER_CONFIG_REG, ((sample_length & 0xffff) << 16) + (sample_start_index & 0xffff) ); // Sample Length
		return_code = vme_crate->vme_A32D32_write (  module_base_addr_array[i_mod] + SIS3316_ADC_CH5_8_RAW_DATA_BUFFER_CONFIG_REG, ((sample_length & 0xffff) << 16) + (sample_start_index & 0xffff) ); // Sample Length
		return_code = vme_crate->vme_A32D32_write (  module_base_addr_array[i_mod] + SIS3316_ADC_CH9_12_RAW_DATA_BUFFER_CONFIG_REG, ((sample_length & 0xffff) << 16) + (sample_start_index & 0xffff) ); // Sample Length
		return_code = vme_crate->vme_A32D32_write (  module_base_addr_array[i_mod] + SIS3316_ADC_CH13_16_RAW_DATA_BUFFER_CONFIG_REG, ((sample_length & 0xffff) << 16) + (sample_start_index & 0xffff) ); // Sample Length
	}

	unsigned int internal_fir_trigger_delay ;
	internal_fir_trigger_delay             =  0 ; 
	
	pre_trigger_delay = pre_trigger_delay + (2 * internal_fir_trigger_delay);   
	// this specifies the amount of event data written to file before the triggering pulse. -m@ 

	//pre_trigger_delay =  0;
	if (pre_trigger_delay > 2042) {
		pre_trigger_delay  = 2042 ;
	}
	//pre_trigger_delay = pre_trigger_delay + 0x8000 ; // set "Additional Delay of Fir Trigger P+G" Bit  
	for (i_mod=0; i_mod<MAX_NOF_SIS3316_ADCS; i_mod++) {
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH1_4_PRE_TRIGGER_DELAY_REG, pre_trigger_delay ); //  
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH5_8_PRE_TRIGGER_DELAY_REG, pre_trigger_delay ); //  
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH9_12_PRE_TRIGGER_DELAY_REG, pre_trigger_delay ); //  
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH13_16_PRE_TRIGGER_DELAY_REG, pre_trigger_delay ); //  
	}

	//pileup ;  
	data = ((uint_re_pileup & 0xffff) << 16) + (uint_pileup & 0xffff) ;
	for (i_mod=0; i_mod<MAX_NOF_SIS3316_ADCS; i_mod++) {
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH1_4_PILEUP_CONFIG_REG, data ); //  
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH5_8_PILEUP_CONFIG_REG, data ); //  
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH9_12_PILEUP_CONFIG_REG, data ); //  
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH13_16_PILEUP_CONFIG_REG, data ); //  
	}


// Enable LEMO Input "TI" as Trigger External Trigger 
	//for (i_mod=0; i_mod<MAX_NOF_SIS3316_ADCS; i_mod++) {
	//	data = 0x10 ; // Enable Nim Input "TI"
	//	return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_NIM_INPUT_CONTROL_REG, data ); //  
	//}

//	unsigned int uint_select_CO_reg ;
//	unsigned int uint_select_TO_reg ;
//	unsigned int uint_select_UO_reg ;

//	uint_select_CO_reg = 0x1 ; // Select Sample Clock
//	uint_select_TO_reg = 0xffff ; // Select all triggers
//	//uint_select_UO_reg = 0x4 ; // Select LogicBusy
//	uint_select_UO_reg = 0x2 ; // Select BankxArmed

//	uint_select_CO_reg = 0x1 ; // Select Sample Clock
	//uint_select_TO_reg = 0xffff ; // Select all triggers
//	uint_select_TO_reg = 0x200000 ; // Select logic armed
	//uint_select_UO_reg = 0x4 ; // Select LogicBusy
	//uint_select_UO_reg = 0x2 ; // Select BankxArmed
//	uint_select_UO_reg = 0x400000 ; //Bank2 flag


//	for (i_mod=0; i_mod<MAX_NOF_SIS3316_ADCS; i_mod++) { 
//		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_LEMO_OUT_CO_SELECT_REG, uint_select_CO_reg ); // Select LEMO Output "CO"    
//		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_LEMO_OUT_TO_SELECT_REG, uint_select_TO_reg ); // Select LEMO Output "TO"  
//		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_LEMO_OUT_UO_SELECT_REG, uint_select_UO_reg ); // Select LEMO Output "UO"  
//	}

//  Event Configuration
	for (i_mod=0; i_mod<MAX_NOF_SIS3316_ADCS; i_mod++) {
	//return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH1_4_EVENT_CONFIG_REG, 0x2020202); // internal sum trigger  
	//return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH1_4_EVENT_CONFIG_REG, 0x0000002); // internal sum trigger  only on ch1
	//return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH1_4_EVENT_CONFIG_REG, 0x04040404 ); // internal trigger ch1 to ch4
	//return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH1_4_EVENT_CONFIG_REG, 0x00000404 ); // internal trigger ch1 to ch2
	//return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH1_4_EVENT_CONFIG_REG, 0x00001515 ); // internal trigger ch1 to ch2 and internal Gate1 enabled (coincidence)

	return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH1_4_EVENT_CONFIG_REG, 0x03030303 ); // internal trigger ch4 , ch3, ch2 ch1 
	//Note that 0x05050505 is only 27 bits, but since the last 5 bits are 0, you don't have to include them -m@
	// I changed the value being written from 0x04040404 -> 0x05050505 inverts the input signal for each channel -m@
	// I changed 0x05050505 to 0x03030303 to switch to 4 channel sum -tja 2013/sep/18
	return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH5_8_EVENT_CONFIG_REG, 0x03030303); // internal trigger ch8 , ch7, ch6 ch5 
	return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH9_12_EVENT_CONFIG_REG, 0x03030303 ); // internal trigger ch12 , ch11, ch10 ch9 
	return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH13_16_EVENT_CONFIG_REG, 0x03030303); // internal trigger ch15 , ch15, ch14 ch13 
	}

	
	maw_test_buffer_delay  = p_val + (p_val>>1) + g_val + 16 + 20 ; // don't know what this calculation actually corresponds to. 
	// TODO 
	if (maw_test_buffer_delay > 1024) {
		maw_test_buffer_delay  = 1024 ;
	}

	if (maw_test_buffer_enable_flag == 0) {
		maw_test_buffer_length =  0 ;
	}


// data format
	header_length = 3;	// this is ok, includes the "number of raw samples" line on p. 48 of 3316 manual 
	header_accu_6_values_offset = 2 ;
	header_accu_2_values_offset = 2 ;
	header_maw_3_values_offset  = 2 ;


	// this section is bascically setting the number off-set lines (i.e. going vertically in the file
	// depending on which types of values we are writting to disk.) So in the first case, the header will be 10 lines long 
	// including the 0xDEADBEEF line that sepearates the header information from the data.--m@ 
	// i think this is correct?? 
	data = 0 ;
	if (header_accu_6_values_enable_flag == 1) { 
		header_length = header_length + 7 ; 
		header_maw_3_values_offset  = header_maw_3_values_offset + 7 ;
		header_accu_2_values_offset  = header_accu_2_values_offset + 7 ;
		data = data + 0x1 ; // set bit 0 //i.e. set Format Bit 0 = 1, -m@ 
	}
	if (header_accu_2_values_enable_flag == 1) {
		header_length = header_length + 2 ;
		header_maw_3_values_offset  = header_maw_3_values_offset + 2 ;
		data = data + 0x2 ; // set bit 1 // i.e Set Format Bit 2 = 1, 0x2 = 10 in binary -m@
	}
	if (header_maw_3_values_enable_flag == 1) {
		header_length = header_length + 3 ;
		data = data + 0x4 ; // set bit 2 // i.e. Set Format Bit 2 = 1, 0x4 = 100 in binary. 
	}
	if (maw_test_buffer_enable_flag == 1) {
		data = data + 0x10 ; // set bit 4 // 0x10 = 1 0000 
		// not sure about this with respect to the info on page 48 
		// should this be 0x8 to set bit 3  
	}
	
	// header_length will be the sum of all of the previous header_lengths: if everything is enabled, it will be 16 lines (as shown on p. 48), including the "number of raw samples" line -m@ 

	event_length = (header_length + (sample_length / 2)  + maw_test_buffer_length); // looking at p. 48, one raw data is only 16 bits, so there are two data per 32 bit line, which is why we divide the sample length by 2... 
	data = data + (data << 8) + (data << 16) + (data << 24);
	
	for (i_mod=0; i_mod<MAX_NOF_SIS3316_ADCS; i_mod++) { 
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH1_4_DATAFORMAT_CONFIG_REG, data ); 
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH5_8_DATAFORMAT_CONFIG_REG, data ); 
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH9_12_DATAFORMAT_CONFIG_REG, data ); 
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH13_16_DATAFORMAT_CONFIG_REG, data ); 
	}

// MAW Test Buffer configuration
	data = maw_test_buffer_length + (maw_test_buffer_delay << 16) ;
	for (i_mod=0; i_mod<MAX_NOF_SIS3316_ADCS; i_mod++) { 
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH1_4_MAW_TEST_BUFFER_CONFIG_REG, data ); 
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH5_8_MAW_TEST_BUFFER_CONFIG_REG, data ); 
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH9_12_MAW_TEST_BUFFER_CONFIG_REG, data ); 
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH13_16_MAW_TEST_BUFFER_CONFIG_REG, data ); 
	}

	//address_threshold = 200;	
	address_threshold = (nof_events * event_length) - 1 ;  //  
	for (i_mod=0; i_mod<MAX_NOF_SIS3316_ADCS; i_mod++) { 
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH1_4_ADDRESS_THRESHOLD_REG, address_threshold ); //  
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH5_8_ADDRESS_THRESHOLD_REG, address_threshold); //   
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH9_12_ADDRESS_THRESHOLD_REG, address_threshold ); //     
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH13_16_ADDRESS_THRESHOLD_REG, address_threshold ); //  
	}

	//by John October 10, 2013
	unsigned int acc_reg=0x0;
	//acc_reg = acc_reg | 0x2;  //strating address
	//acc_reg = acc_reg | (0xa << 16); //length
	acc_reg = 0x00090000;

	for (i_mod=0; i_mod<MAX_NOF_SIS3316_ADCS; i_mod++) {
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH1_4_ACCUMULATOR_GATE1_CONFIG_REG, acc_reg ); //
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH5_8_ACCUMULATOR_GATE1_CONFIG_REG, acc_reg ); //
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH9_12_ACCUMULATOR_GATE1_CONFIG_REG, acc_reg ); //
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH13_16_ACCUMULATOR_GATE1_CONFIG_REG, acc_reg ); //
	}

	acc_reg = 0x0;
	acc_reg = 0x001D000F; // 1D = 29 = 30 samples
	for (i_mod=0; i_mod<MAX_NOF_SIS3316_ADCS; i_mod++) {
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH1_4_ACCUMULATOR_GATE2_CONFIG_REG, acc_reg ); //
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH5_8_ACCUMULATOR_GATE2_CONFIG_REG, acc_reg ); //
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH9_12_ACCUMULATOR_GATE2_CONFIG_REG, acc_reg ); //
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ADC_CH13_16_ACCUMULATOR_GATE2_CONFIG_REG, acc_reg ); //
	}

	loop_counter=0;
	timeout_counter = 0 ;
	printf("Start Multievent \n");
	
	gl_graph_raw->sis3316_draw_XYaxis (sample_length); // clear and draw X/Y
	//gl_graph_raw->sis3316_draw_XYaxis (20); // clear and draw X/Y

	peak_test_error_counter = 0 ;
	gate_test_error_counter = 0 ;
	maw_value_error_counter = 0 ;

	unsigned int bank_buffer_counter ;
	bank_buffer_counter = 0 ; //this gets incremented each time there is a buffer read-out -m@ 


	// file write
	//cout << "Write data to file? (1 for yes, 0 for no): "; //-m@
	//cin >> uint_DataEvent_OpenFlag;  
	uint_DataEvent_OpenFlag = 1 ;
	


	char mystr[80]; //-m@
	if (uint_DataEvent_OpenFlag == 1) {   ; // Open
		//cout << "File Name (include .dat extention): " ; 
		//cin >> mystr;  //-m@ 
		
		// names data file with current timestamp
		std::time_t rawtime;
		std::tm* timeinfo;
		std::time(&rawtime);
		timeinfo = std::localtime(&rawtime);
		std::strftime(mystr,80,"%Y-%m-%d-%H-%M-%S",timeinfo);
		std::puts(mystr);
		strcat(mystr,".dat");
		gl_FILE_DataEvenFilePointer = fopen(mystr,"wb") ; //
        
        WriteDataFileHeader(header_length,sample_length, maw_test_buffer_length); 
	}



	// enbale external (global) functions
	//data = 0x100 ; // enable "external Trigger function" (NIM In, if enabled and VME key write)
	//data = data + 0x400 ; // enable "external Timestamp clear function" (NIM In, if enabled and VME key write)
	data = 0x40 ; // enable "FP-Bus-In Timestamp clear function" 
	data = data + 0x80 ; // enable "FP-Bus-In Sample Control"
	for (i_mod=0; i_mod<MAX_NOF_SIS3316_ADCS; i_mod++) { 
		return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[i_mod] + SIS3316_ACQUISITION_CONTROL_STATUS, data );  
	}


	// Clear Timestamp  */
	// Clear Timestamp on first module --> will clear via FB-Bus Timestamps on all modules */
	return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[0] + SIS3316_KEY_TIMESTAMP_CLEAR , 0);  
	

		
	// Start Readout Loop  */
	return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[0] + SIS3316_KEY_DISARM_AND_ARM_BANK1 , 0);  //  Arm Bank1 
	
	bank1_armed_flag = 1; // start condition
	printf("SIS3316_KEY_DISARM_AND_ARM_BANK1 \n");

	
	unsigned int event_counter_bank;
	unsigned int plot_counter;
	unsigned int MaxBufferReads; 
	plot_counter=0;
	//cout <<"Number of buffer reads (the buffer writes out about every 0.5 sec, so 120 corresponds to a collection time of approximately 1 min.) : "; 
	//cin >> MaxBufferReads; 
  //-m@  --the buffer writes out about every 0.5 sec, not sure how to get a very precise measurement for this, this is only approximate. 
	
	double timer; // jfr - 17aug15 changed from int to double
	int ticks;
	clock_t time_start;
	clock_t time_now;

	cout << "Enter time of data aquisition in min.: ";
	cin >> timer;
	// timer = .5; //75
	cout << "Aquisition time set to " << timer << " min. "; // jfr - 27jan15
	cout << endl;
	ticks = CLOCKS_PER_SEC * timer * 60;

	time_start = clock();

	do {
		poll_counter = 0 ;
		do {
			poll_counter++;
			if (poll_counter == 100) {
				gSystem->ProcessEvents();  // handle GUI events
				poll_counter = 0 ;
    		}
			vme_crate->vme_A32D32_read ( module_base_addr_array[0] + SIS3316_ACQUISITION_CONTROL_STATUS, &data);  
			
			
			//usleep(500000); //500ms  
			//printf("in Loop:   SIS3316_ACQUISITION_CONTROL_STATUS = 0x%08x     \n", data);
		//} while ((data & 0x80000) == 0x0) ; // Address Threshold reached ?
		} while ((data & 0x200000) == 0x0) ;  // FP-Bus Address Threshold reached ? 

		
		if (bank1_armed_flag == 1) {		
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[0] + SIS3316_KEY_DISARM_AND_ARM_BANK2 , 0);  //  Arm Bank2
			
			bank1_armed_flag = 0; // bank 2 is armed
			printf("SIS3316_KEY_DISARM_AND_ARM_BANK2 \n");
		}
		else {
			return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[0] + SIS3316_KEY_DISARM_AND_ARM_BANK1 , 0);  //  Arm Bank1
			
			
			bank1_armed_flag = 1; // bank 1 is armed
			printf("SIS3316_KEY_DISARM_AND_ARM_BANK1 \n");
		}

		printf("header_length=%d, event length=%d\n", header_length, event_length);

		event_counter_bank = 0;

		gl_graph_raw->sis3316_draw_XYaxis (sample_length); // clear and draw X/Y
		for (i_mod=0; i_mod<MAX_NOF_SIS3316_ADCS; i_mod++) {
			for (i_ch=0; i_ch<16; i_ch++) {
				// read channel events 
				return_code = sis3316_adc_array[i_mod]->read_MBLT64_Channel_PreviousBankDataBuffer(bank1_armed_flag /*bank2_read_flag*/, i_ch /* 0 to 15 */,  &got_nof_32bit_words, gl_rblt_data ) ;
				// this just reads the data from bank 2 for each channnel (stored in array gl_rblt_data) and returns the number of samples: got_nof_32bit_words -m@

				printf("read_MBLT64_Channel_PreviousBankDataBuffer: i_ch %d  got_nof_32bit_words = 0x%08x  return_code = 0x%08x\n",i_ch,  got_nof_32bit_words, return_code);
				//if (return_code != 0) {
					//printf("read_MBLT64_Channel_PreviousBankDataBuffer: return_code = 0x%08x\n", return_code);
					//gl_stopReq = TRUE;
				//}
				ch_event_counter = (got_nof_32bit_words  / event_length) ; //# events per channel = total # of samples/event length (same for each channel) , something like that -m@ 
			
				if (ch_event_counter > 0) {
					event_counter_bank = event_counter_bank + ch_event_counter;
					// plot events
					for (i=0; i<ch_event_counter; i++) {
						if (i==0) { // plot ony 1. event
							gl_graph_raw->sis3316_draw_chN (sample_length, &gl_rblt_data[i*(event_length) + header_length], i_ch); //  
						}
					}
					if (uint_DataEvent_OpenFlag == 1) {   ; // Open
						WriteBufferHeaderCounterNofChannelToDataFile (bank_buffer_counter, ch_event_counter , got_nof_32bit_words) ; 
						//This header information will be neccessary for proper reading of the events. 
						WriteEventsToDataFile (gl_rblt_data, got_nof_32bit_words)  ; //write data to file TODO 

						/// ** THIS DOESN"T INCLUDE THE CHANNEL NUMBER, MAY NEED TO INCLUDE THIS!!!, i.e. if ch_event_counter =0, then it skips writting and iterates on to the next channel, so there is no way to tell in the data file which data corresponds to which channel  -m@

						//actually, each "channel" event 
					}
    			}
			}
			//added readout of ADC-FPGA status register for diagnostics 21apr14 jfr
			vme_crate->vme_A32D32_read ( module_base_addr_array[i_mod] + SIS3316_ADC_CH1_4_STATUS_REG, &data);  
			printf("SIS3316_ADC_CH1_4_STATUS_REG     = 0x%08x \n", data);

			vme_crate->vme_A32D32_read ( module_base_addr_array[i_mod] + SIS3316_ADC_CH5_8_STATUS_REG, &data);  
			printf("SIS3316_ADC_CH5_8_STATUS_REG     = 0x%08x \n", data);

			vme_crate->vme_A32D32_read ( module_base_addr_array[i_mod] + SIS3316_ADC_CH9_12_STATUS_REG, &data);  
			printf("SIS3316_ADC_CH9_12_STATUS_REG    = 0x%08x \n", data);

			vme_crate->vme_A32D32_read ( module_base_addr_array[i_mod] + SIS3316_ADC_CH13_16_STATUS_REG, &data);  
			printf("SIS3316_ADC_CH13_16_STATUS_REG   = 0x%08x \n\n", data);

			//added a print temperature statement for diagnostics --jrf 21Apr2014
			vme_crate->vme_A32D32_read ( module_base_addr + SIS3316_INTERNAL_TEMPERATURE_REG, &data);
			printf("ADC temperature = %08x \n\n", data); 
			
		}



		loop_counter++;
		//usleep(500); //unsigned int uint_usec  
		usleep(500000); //500ms  

		bank_buffer_counter++; //incrementation each time there is a buffer read out -m@
		
		printf("bank_buffer_counter = %d     \n",bank_buffer_counter);
		printf("ch_event_counter    = %d     \n", ch_event_counter);
		gSystem->ProcessEvents();  // handle GUI events

		time_now = clock() - time_start;

	} while(( time_now <= ticks));         //(gl_stopReq == FALSE)); //bank_buffer_counter<MaxBufferReads) ); -m@ 
    
	cout << endl << "Aquisition complete" << endl; // jfr 2013sep19

	if (uint_DataEvent_OpenFlag == 1) {   // Open
		WriteEOF(); // -m@ 
        fclose(gl_FILE_DataEvenFilePointer);
	}
	return_code = vme_crate->vme_A32D32_write ( module_base_addr_array[0] + SIS3316_KEY_DISARM , 0);  //   
	
	theApp.Terminate(); // jfr 27jan15

	gl_stopReq = FALSE;
	unsigned int buffer_counter = 0; 
	do {
		gSystem->ProcessEvents(); 
		buffer_counter++; // handle GUI events
	} while(( gl_stopReq == FALSE));//buffer_counter<bank_buffer_counter) ); -m@

	
	return 0;
}

/***********************************************************************************************************************************************/
/***********************************************************************************************************************************************/
/***********************************************************************************************************************************************/

//TODO: 
#define FILE_FORMAT_EVENT_HEADER        	0xDEADBEEF  
#define FILE_FORMAT_EOF_TRAILER        		0x0E0F0E0F 
#define DATA_FILE_HEADER                    0xABABABAB

int WriteDataFileHeader(unsigned int headerlength, unsigned int samplelength, unsigned int mawtestbufferlength) ///added by -m@
{
    
	int written;
    int data;
    int header_length_read;
    int sample_length_read;
    int maw_test_buffer_length_read;
    
    data = DATA_FILE_HEADER;
    header_length_read = headerlength;
    sample_length_read = samplelength;
    maw_test_buffer_length_read = mawtestbufferlength;
    
    written = fwrite(&data,0x4,0x1,gl_FILE_DataEvenFilePointer);
    written = fwrite(&header_length_read,0x4,0x1,gl_FILE_DataEvenFilePointer); 
    written = fwrite(&sample_length_read,0x4,0x1,gl_FILE_DataEvenFilePointer);
    written = fwrite(&maw_test_buffer_length_read,0x4,0x1,gl_FILE_DataEvenFilePointer);
     
    
    return 0; 
}

//---------------------------------------------------------------------------------------

int WriteEOF()
{
    int written;
    int data;
    
    
    data = FILE_FORMAT_EOF_TRAILER;
   
    
    written = fwrite(&data, 0x4, 0x1,gl_FILE_DataEvenFilePointer);
    
    return 0; 
    
}

//---------------------------------------------------------------------------------------

int WriteBufferHeaderCounterNofChannelToDataFile (unsigned int buffer_no,unsigned int nof_events, unsigned int event_length)
{   
int written ;
int data ;
 
// Each of the following will take up 32 bits, i.e. 1 "line" of the header 
    
    //header
	data = FILE_FORMAT_EVENT_HEADER ;    
    written=fwrite(&data,0x4,0x1,gl_FILE_DataEvenFilePointer); // write one  uint word
	//gl_uint_DataEvent_LWordCounter = gl_uint_DataEvent_LWordCounter + written ;
  
    //Buffer No
    written=fwrite(&buffer_no,0x4,0x1,gl_FILE_DataEvenFilePointer); // write one  uint word
	//gl_uint_DataEvent_LWordCounter = gl_uint_DataEvent_LWordCounter + written ;
  
    //nof events
    written=fwrite(&nof_events,0x4,0x1,gl_FILE_DataEvenFilePointer); // write one  uint word
	//gl_uint_DataEvent_LWordCounter = gl_uint_DataEvent_LWordCounter + written ;
  
    //event length
    written=fwrite(&event_length,0x4,0x1,gl_FILE_DataEvenFilePointer); // write one  uint word
	
	
	//gl_uint_DataEvent_LWordCounter = gl_uint_DataEvent_LWordCounter + written ;
	//gl_uint_DataEvent_RunFile_EventChannelSize =  event_length;
	//gl_uint_DataEvent_RunFile_EventSize = nof_channels * gl_uint_DataEvent_RunFile_EventChannelSize ;
 	
 	return 0;

}


//--------------------------------------------------------------------------- 
int WriteEventsToDataFile (unsigned int* memory_data_array, unsigned int nof_write_length_lwords)
{   
int nof_write_elements ;
int written ;
int data ;
char messages_buffer[256] ;           

// gl_uint_DataEvent_RunFile_EvenSize : length 

		nof_write_elements = nof_write_length_lwords ;
		written=fwrite(memory_data_array,0x4,nof_write_elements,gl_FILE_DataEvenFilePointer); // write 3 uint value
		//gl_uint_DataEvent_LWordCounter = gl_uint_DataEvent_LWordCounter + written  ;
		if(nof_write_elements != written) { 
    		printf ("Data File Write Error in  WriteEventToDataFile()  \n");
 		 }

 	return 0;

}


#ifdef raus
	if (uint_DataEvent_OpenFlag == 1) {   ; // Open
		WriteBufferHeaderCounterNofChannelToDataFile (buffer_switch_counter, nof_events, event_length_lwords) ;
		WriteEventsToDataFile (gl_dma_rd_buffer, dma_got_no_of_words)  ;
	}

FILE *gl_FILE_DataEvenFilePointer           ;
int written ;
unsigned int gl_uint_DataEvent_FileCounter ;

	gl_uint_DataEvent_FileCounter = 0x1 ;
     written=fwrite(&gl_uint_DataEvent_FileCounter,0x4,0x1,gl_FILE_DataEvenFilePointer); // write one  uint word
     written=fwrite(&gl_uint_DataEvent_FileCounter,0x4,0x1,gl_FILE_DataEvenFilePointer); // write one  uint word
     written=fwrite(&gl_uint_DataEvent_FileCounter,0x4,0x1,gl_FILE_DataEvenFilePointer); // write one  uint word
     written=fwrite(&gl_uint_DataEvent_FileCounter,0x4,0x1,gl_FILE_DataEvenFilePointer); // write one  uint word
     written=fwrite(&gl_uint_DataEvent_FileCounter,0x4,0x1,gl_FILE_DataEvenFilePointer); // write one  uint word
     written=fwrite(&gl_uint_DataEvent_FileCounter,0x4,0x1,gl_FILE_DataEvenFilePointer); // write one  uint word
     written=fwrite(&gl_uint_DataEvent_FileCounter,0x4,0x1,gl_FILE_DataEvenFilePointer); // write one  uint word
     written=fwrite(&gl_uint_DataEvent_FileCounter,0x4,0x1,gl_FILE_DataEvenFilePointer); // write one  uint word
     written=fwrite(&gl_uint_DataEvent_FileCounter,0x4,0x1,gl_FILE_DataEvenFilePointer); // write one  uint word
    fclose(gl_FILE_DataEvenFilePointer);
#endif
/***********************************************************************************************************************************************/
/***********************************************************************************************************************************************/
/***********************************************************************************************************************************************/


void program_stop_and_wait(void)
{
	gl_stopReq = FALSE;
	printf( "\n\nProgram stopped");
	printf( "\n\nEnter ctrl C");
	do {
		Sleep(1) ;
	} while (gl_stopReq == FALSE) ;
	
	//		result = scanf( "%s", line_in );
}


/***************************************************/


BOOL CtrlHandler( DWORD ctrlType ){
	switch( ctrlType ){
	case CTRL_C_EVENT:
		printf( "\n\nCTRL-C pressed. finishing current task.\n\n");
		gl_stopReq = TRUE;
		return( TRUE );
		break;
	default:
		printf( "\n\ndefault pressed. \n\n");
		return( FALSE );
		break;
	}
}



//#ifdef WINDOWS
void usleep(unsigned int uint_usec) 
{
    unsigned int msec;
	if (uint_usec <= 1000) {
		msec = 1 ;
	}
	else {
		msec = (uint_usec+999) / 1000 ;
	}
	Sleep(msec);

}
//#endif

