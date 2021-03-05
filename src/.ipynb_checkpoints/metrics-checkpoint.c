
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <math.h>
#include <windows.h>
#include "resource.h"
#include "globals.h"

		/*
		** Calculates time-based metrics (per second) and event-based metrics (per meal).
		** The input structure contains the segments.  This function fills in the results fields.
		*/

void CalculateResults(struct results *classifier)

{
int			e,i,t;
int			*GTEval,*MSEval;
int			total_sec,gt_ea_sec,gt_non_ea_sec;	/* total seconds of valid data, time spent eating, and time not eating */
int			md_ea_sec,md_non_ea_sec;			/* total seconds labeled by the classifer as eating and not eating */
int			tp_sec,tn_sec,fp_sec,fn_sec;		/* total #seconds correctly classified as eating (tp), as not eating (tn), */
												/* and incorrectly classified as eating when not eating (fp) and not eating when eating (fn) */
int			data_valid,data_unclassified;		/* total indices of valid data and data not classified (should be zero) */
int			data_eating_gt;						/* total indices of data self-reported as EA */
int			data_eating_tp,data_eating_fp;		/* total indices classified as EA, both TP and FP */
int			TPs,FPs,Missed;						/* event-level metrics */
int			boundary_error_start,boundary_error_end;  /* boundary errors for event detections */
int			exact_match;						/* 0=> +-1 hour is a match (used for single time event reports); 1=> must overlap */
int			distance1,distance2;				/* for inexact matches, used to find closest GT event */

		/* initialize all GT and MS (potential EA detections) as un-evaluated */
GTEval=(int *)calloc(MAX_SEG,sizeof(int));
MSEval=(int *)calloc(MAX_SEG,sizeof(int));
for (i=0; i<TotalEvents; i++)
  GTEval[i]=-1;
for (e=0; e<classifier->TotalSpans; e++)
  if (classifier->classified[e] != 0)	  /* span not classified as eating - do not try to match to GT event */
	MSEval[e]=-2;						  /* identify here in case there are zero GT events */
  else
	MSEval[e]=-1;						  /* look for match with GT event */
for (i=0; i<TotalEvents; i++)
  {
  if (EventEnd[i]-EventStart[i] < 60*15)
	exact_match=0;		  /* event is 59 sec or less, indicating only a single time (to the minute) was reported in GT */
  else
	exact_match=1;
  for (e=0; e<classifier->TotalSpans; e++)
    {
	if (MSEval[e] == -2)
	  continue;			  /* not classified as eating */
	if (exact_match == 1)
	  {
			/* test 4 possibilities:  (1) MD within GT (2) MD starts before GT and ends in GT */
			/* (3) MD starts in GT and ends after GT (4) MD contains GT */
	  if ((classifier->Spans[e][0] >= EventStart[i]  &&  classifier->Spans[e][1] <= EventEnd[i])  ||
		  (classifier->Spans[e][0] <= EventStart[i]  &&  classifier->Spans[e][1] >= EventStart[i]  &&
				  classifier->Spans[e][1] <= EventEnd[i])  ||
		  (classifier->Spans[e][0] >= EventStart[i]  &&  classifier->Spans[e][0] <= EventEnd[i]  &&
				  classifier->Spans[e][1] >= EventEnd[i])  ||
		  (classifier->Spans[e][0] <= EventStart[i]  &&  classifier->Spans[e][1] >= EventEnd[i]))
		{		/* any overlap counts as a detection */
		GTEval[i]=e;
		MSEval[e]=i;
		boundary_error_start=classifier->Spans[GTEval[i]][0]-EventStart[i];
		boundary_error_end=classifier->Spans[GTEval[i]][1]-EventEnd[i];
		}
	  }
	else
	  {
	  if (MSEval[e] != -1)
		{	/* already matched one GT event, check if this new match is closer */
		distance1=((EventStart[MSEval[e]]+EventEnd[MSEval[e]])/2)-((classifier->Spans[e][0]+classifier->Spans[e][1])/2);
		distance2=((EventStart[i]+EventEnd[i])/2)-((classifier->Spans[e][0]+classifier->Spans[e][1])/2);
		if (abs(distance1) < abs(distance2))
		  continue;	  /* previous matching GT event is closer */
		}
			/* test if the MD start or MD end is within 1 hour of the GT event start */
			/* allows multiple MDs to connect to a single GT */
	  if (abs((classifier->Spans[e][0])-EventStart[i]) <= 60*60*15  ||
		  abs((classifier->Spans[e][1])-EventStart[i]) <= 60*60*15)
		{		/* within 1 hour counts as a detection */
		GTEval[i]=e;
		MSEval[e]=i;
		boundary_error_start=classifier->Spans[GTEval[i]][0]-EventStart[i];
		boundary_error_end=classifier->Spans[GTEval[i]][1]-EventEnd[i];
		}
	  }
    }
  }
		/* count up correct detections, false positives and misses */
TPs=FPs=Missed=0;
for (i=0; i<TotalEvents; i++)
  {
  if (GTEval[i] == -1)
	Missed++;
  else
	TPs++;
  }
for (e=0; e<classifier->TotalSpans; e++)
  if (MSEval[e] == -1)
	FPs++;

		/* calculate accuracies in seconds */
t=0;					/* datum index */
data_valid=0;
data_unclassified=0;
data_eating_gt=0;
data_eating_tp=0;
data_eating_fp=0;

while (t<Total_Data)
  {
  if (motion_state[t] == 0)
	{
	t++;
	continue;			/* invalid data (device off) */
	}
  data_valid++;
		/* determine if datum inside a self-reported EA (GT event) */
  for (i=0; i<TotalEvents; i++)
	if (t >= EventStart[i]  &&  t <= EventEnd[i])
	  break;
		/* count up self-report GT */
  if (i < TotalEvents)
	data_eating_gt++;
 		/* determine segment containing datum */
  for (e=0; e<classifier->TotalSpans; e++)
	if (t >= classifier->Spans[e][0]  &&  t <= classifier->Spans[e][1])
	  break;
  if (e == classifier->TotalSpans)
	{
	data_unclassified++;
	t++;
	continue;
	}
		/* count machine detected intakes (TPs and FPs) */
  if (classifier->classified[e] == 0)
	{
	if (i < TotalEvents)
	  data_eating_tp++;
	else
	  data_eating_fp++;
	}
  t++;
  }

total_sec=data_valid/15;
gt_ea_sec=data_eating_gt/15;
gt_non_ea_sec=total_sec-gt_ea_sec;
md_ea_sec=(data_eating_tp+data_eating_fp)/15;
md_non_ea_sec=total_sec-md_ea_sec;
tp_sec=data_eating_tp/15;
fn_sec=gt_ea_sec-tp_sec;
fp_sec=md_ea_sec-tp_sec;
tn_sec=gt_non_ea_sec-fp_sec;

		/* global vars for all classifiers */
TimeRecorded=total_sec;
TimeEating=gt_ea_sec;
		/* vars specific to this classifier */
classifier->TP_sec=tp_sec;
classifier->TN_sec=tn_sec;
classifier->WeightedAccuracy=(((double)tp_sec)*20.0+(double)tn_sec)/(((double)gt_ea_sec)*20.0+(double)gt_non_ea_sec);
classifier->TPs=TPs;
classifier->FPs=FPs;
classifier->Missed=Missed;

free(GTEval);
free(MSEval);
}


