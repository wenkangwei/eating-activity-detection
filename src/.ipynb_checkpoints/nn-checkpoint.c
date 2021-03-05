
	/*
	** These functions process a data file and use a
	** TensorFlow neural network to detect periods of eating.
	*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <tensorflow/c/c_api.h>
#include <windows.h>
#include "globals.h"


	/* parameters for neural network classifier */
#define	WINDOW_SIZE	5400  /* 5400 = 6 minutes @ 15 Hz, 9000 = 10 minutes */
#define	WINDOW_STEP	1500  /* 100 sec @ 15 Hz */
	/* parameters for eating detector (analyzing NN output) */
#define	T_HIGH_DEF	0.8	/* hysteresis high threshold */
#define	T_LOW_DEF	0.3	/* hysteresis low threshold */
#define	T_PAUSE		900	/* #indices to check eating not resumed */
#define	T_MIN		900	/* #indices minimum eating episode */

#define	NORM_WINDOW	0	/* 0=>per file, 1=>per window */
#define	NORM_METHOD	0	/* 0=>z-score, 1=>min/max normalize */
				/* note: only implemented 0/0 and 1/1 so far */


void TensorFlowRun()
{
float	*windowdata;
double	min[6],max[6];
int		i,j,k;
float	last_update;
int		start,end,state;
float	eating_prob;
int		pause_counter;
int		resting,valid_data;
double	shimmer_global_mean[6]={-0.012359981,-0.0051663737,0.011612018, \
				0.05796114,0.1477952,-0.034395125};
//double	shimmer_global_mean[6]={0.0,0.0,0.0,0.0,0.0,0.0};
double	shimmer_global_stddev[6]={0.05756385,0.040893298,0.043825723, \
				17.199743,15.311142,21.229317};

	/* allocate space for data */
windowdata=(float *)calloc(WINDOW_SIZE*6,sizeof(float)); // input to NN

i=0;
state=0;				  // 0=>not eating, 1=>eating, 2=>waiting for possible resume
last_update=0.0;		  // last time (percentage of total data) window refreshed
NN_Results.TotalSpans=0;  // # of detected eating segments
while (i < Total_Data-WINDOW_SIZE)
  {
  if (NORM_WINDOW == 1)
    {	/* note -- only 1/1 or 0/0 implemented so far */
    if (NORM_METHOD == 1)
      {	/* find min/max per sensor row per window to normalize */
      for (k=0; k<6; k++)
        {
        if (i < 0)
          min[k]=max[k]=Smoothed[k][0];
        else
          min[k]=max[k]=Smoothed[k][i];
        for (j=0; j<WINDOW_SIZE; j++)
          {
          if (i+j < 0  ||  i+j >= Total_Data)
			continue;
          if (Smoothed[k][i+j] < min[k])
			min[k]=Smoothed[k][i+j];
          if (Smoothed[k][i+j] > max[k])
			max[k]=Smoothed[k][i+j];
          }
        if (max[k]-min[k] < 0.00001)	// prevent divide by zero
          max[k]+=0.00001;	// all #s will normalize to zero
        }
      }
    }
	/* window data[] is a 1D array and is organized as follows:
	** t0[x y z Y P R]  t1[x y z Y P R]  t2[x y z Y P R] ... */
  for (j=0; j<WINDOW_SIZE; j++)
    {
    if (i+j < 0  ||  i+j >= Total_Data)
      {
      for (k=0; k<6; k++)
		windowdata[j*6+k]=0.0;
      }
    else
      {
      for (k=0; k<6; k++)
        {
        if (NORM_WINDOW == 0)
          windowdata[j*6+k]=(float)((Smoothed[k][i+j]-shimmer_global_mean[k])/shimmer_global_stddev[k]);
        else if (NORM_METHOD == 1)
          windowdata[j*6+k]=(float)((Smoothed[k][i+j]-min[k])/(max[k]-min[k]));
        }
      }
    }

		/* calculate percentage of window that data is at rest */
  resting=valid_data=0;
  for (j=0; j<WINDOW_SIZE; j++)
	{
	if (motion_state[i+j] == 1)
	  resting++;
	if (motion_state[i+j] != 0)
	  valid_data++;
	}
  if ((double)resting/(double)valid_data > 0.65)
	eating_prob=0.0;
  else
	eating_prob=TensorFlowClassify(windowdata);
  nn_prob[i]=eating_prob;
  for (j=i+1; j<i+WINDOW_STEP; j++)
	nn_prob[j]=nn_prob[i];
  
	/* wait for percentage to go above high threshold */
  if (state == 0  &&  eating_prob > nn_t_high)
    {
    state=1;
    start=i;
    }
  if (state == 1  &&  eating_prob < nn_t_low)
    {
    state=2;
    end=i+1;
    pause_counter=0;
    }
  if (state == 2)
    {
    if (eating_prob > nn_t_high)
      state=1;	// user resumed eating
    else
      {
      pause_counter+=WINDOW_STEP;
      if (pause_counter >= T_PAUSE)
        {	// detect eating episode
        if (end-start > T_MIN)	// must be 1 minute minimum
          {
		  NN_Results.Spans[NN_Results.TotalSpans][0]=start; //+WINDOW_SIZE/2;
		  NN_Results.Spans[NN_Results.TotalSpans][1]=end; //+WINDOW_SIZE/2;
		  NN_Results.classified[NN_Results.TotalSpans]=0; // EA
		  (NN_Results.TotalSpans)++;
          }
        state=0;
		end=0;	// reset so can be used for end-of-data meal
        }
      }
    }

  i+=WINDOW_STEP;
  if ((float)i/(float)(Total_Data-WINDOW_SIZE) > last_update+0.1)	// every 10% redraw
	{
	PaintImage();
	last_update=(float)i/(float)(Total_Data-WINDOW_SIZE);
	}
  }
if (state != 0)
  {	  // was in the middle of detecting a meal when data ended
  NN_Results.Spans[NN_Results.TotalSpans][0]=start;
  if (end != 0)
	NN_Results.Spans[NN_Results.TotalSpans][1]=end;
  else
	NN_Results.Spans[NN_Results.TotalSpans][1]=Total_Data-1;
  NN_Results.TotalSpans++;
  }
PaintImage();

	/* release memory */
free(windowdata);
}




	/* for TF_NewBuffer, used in reading the file */
void free_buffer(void* data, size_t length) { free(data); }

#define INPUT_LENGTH	WINDOW_SIZE	/* amount of data per input pattern */
#define	INPUT_AXES	6		/* number of axes per input pattern */
#define OUTPUT_LENGTH	1		/* number of classes in output */
					/* eating(1) or not(0) */

	/*
	** The input and output layer names can be found by placing the
	** following 2 lines in the python program used for training:
	** print('model inputs: ',model.inputs)
	** print('model outputs: ',model.outputs)
	** Put the printed strings into the below two variables.
	*/
char	*input_layer_name="conv1d_1_input";
char	*output_layer_name="dense_2/Sigmoid";

	/* TF library stuff */
static TF_Graph				*graph;
static TF_Status			*status;
static TF_ImportGraphDefOptions	*graph_opts;
static TF_SessionOptions 		*session_opts;
static TF_Buffer			*graph_def;
static TF_Session			*session;
static TF_Operation			*input_layer,*output_layer;
static TF_Tensor			*input_tensor,*output_tensor;
static TF_Output			network_inputs,network_outputs;


	/* load the TF model, initialize the TF classifier */

void TensorFlowInit()

{
FILE	*fpt;
long	fsize;
void	*data;
int64_t	input_dim[3]={1,INPUT_LENGTH,INPUT_AXES};
int64_t	output_dim[2]={1,OUTPUT_LENGTH};
char	model_filename[320];

	
	/*
	** filename for the PB mdoel (note this program
	** assumes a stripped down PB file, no tags in it)
	*/
if (WINDOW_SIZE == 9000)
  sprintf(model_filename,"Watch10min.pb");
else
  sprintf(model_filename,"NewWatch6min.pb");
graph=TF_NewGraph();	/* holds model */
status=TF_NewStatus();	/* can be checked for errors after every TF call */
	/* read graph from file */
if ((fpt=fopen(model_filename,"rb")) == NULL)
  {
  printf("Unable to open %s for reading\n",model_filename);
  exit(0);
  }
fseek(fpt,0,SEEK_END);
fsize=ftell(fpt);
fseek(fpt,0,SEEK_SET);
data=malloc(fsize);
fread(data,fsize,1,fpt);
fclose(fpt);
graph_def=TF_NewBuffer();
graph_def->data=data;
graph_def->length=fsize;
graph_def->data_deallocator=free_buffer;	/* used in TF_DeleteGraph() */
graph_opts=TF_NewImportGraphDefOptions();
TF_GraphImportGraphDef(graph,graph_def,graph_opts,status);
TF_DeleteImportGraphDefOptions(graph_opts);
if (TF_GetCode(status) != TF_OK)
  {
  printf("Unable to import graph from %s\n",model_filename);
  exit(0);
  }

	/* make a session (instantiate the model) */
session_opts=TF_NewSessionOptions();
session=TF_NewSession(graph,session_opts,status);
TF_DeleteSessionOptions(session_opts);
if (TF_GetCode(status) != TF_OK)
  {
  printf("Unable to create session\n");
  exit(0);
  }

	/* find input layer */
input_layer=TF_GraphOperationByName(graph,input_layer_name);
if (input_layer == NULL)
  {
  printf("input layer name %s not found\n",input_layer_name);
  exit(0);
  }
network_inputs.oper=input_layer;
network_inputs.index=0;

	/* find output layer */
output_layer=TF_GraphOperationByName(graph,output_layer_name);
if (output_layer == NULL)
  {
  printf("output layer name %s not found\n",output_layer_name);
  exit(0);
  }
network_outputs.oper=output_layer;
network_outputs.index=0;

	/* create tensors, get pointers to data buffers inside them */
input_tensor=TF_AllocateTensor(TF_FLOAT,input_dim,3,
		INPUT_LENGTH*INPUT_AXES*sizeof(float));
output_tensor=TF_AllocateTensor(TF_FLOAT,output_dim,2,
		OUTPUT_LENGTH*sizeof(float));
}




	/* classify one window of data, return #bites in the window */

float TensorFlowClassify(float *sampledata)

{
int		i;
float	*input_buf_ptr,*output_buf_ptr;	/* access to data in tensors */
float	ret;


	/* copy data into input tensor */
input_buf_ptr=TF_TensorData(input_tensor);
for (i=0; i<INPUT_LENGTH*INPUT_AXES; i++)
  {
  input_buf_ptr[i]=sampledata[i];
  }

	/* perform classification (run the neural network) */
TF_SessionRun(session,
	NULL,					/* run options */
	&network_inputs,&input_tensor,1,	/* input tensor(s) */
	&network_outputs,&output_tensor,1,	/* output tensor(s) */
	NULL,0,					/* target op(s) */
	NULL,					/* run metadata */
	status);				/* output status */
if (TF_GetCode(status) != TF_OK)
  {
  printf("Unable to perform classification: %s\n", TF_Message(status));
  exit(0);
  }

	/* get result from output tensor */
output_buf_ptr=TF_TensorData(output_tensor);	/* get the new ptr */
ret=output_buf_ptr[0];
	/*
	** TF_SessionRun() allocates a new tensor for the output.
	** It must be deleted to prevent a memory leak.  See c_api.h
	*/
TF_DeleteTensor((TF_Tensor *)output_tensor);
return(ret);
}




	/* free the TF variables created for the classifier */

void TensorFlowCleanup()

{
TF_CloseSession(session, status);
TF_DeleteSession(session, status);
TF_DeleteStatus(status);
TF_DeleteBuffer(graph_def);
TF_DeleteGraph(graph);
}


