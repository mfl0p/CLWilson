/*
	CLWilson
	Bryan Little, July 2025
	
	with contributions by Yves Gallot and Kim Walisch

	Required minimum OpenCL version is 1.1
	CL_TARGET_OPENCL_VERSION 110 in simpleCL.h

	Current search limits:  
	-p from 5, due to splitting of primes into 3 types
	-P up to (2^64-1)/4 = 4611686018427387903, due to overflow during find_c and find_u

*/

#include <unistd.h>
#include <cinttypes>
#include <math.h>
#include <algorithm>

#ifdef _WIN32
  #include "gmpwin.h"
#else
  #include "gmp.h"
#endif

#include "boinc_api.h"
#include "boinc_opencl.h"
#include "simpleCL.h"

#include "clearn.h"
#include "clearresult.h"
#include "getsegprps.h"
#include "setup.h"
#include "iterate.h"
#include "mulsmall.h"
#include "mullarge.h"
#include "reduce.h"
#include "find.h"
#include "common.h"

#include "primesieve.h"
#include "putil.h"
#include "cl_wilson.h"

#define RESULT_FILENAME "results.txt"
#define STATE_FILENAME_A "stateA.ckp"
#define STATE_FILENAME_B "stateB.ckp"

#if __LDBL_MANT_DIG__ < 64
#error Long Double Mantissa is too small
#endif

#define ACUBUFFER 100
#define PRPSIZE 12446226

void handle_trickle_up(workStatus & st){
	if(boinc_is_standalone()) return;
	uint64_t now = (uint64_t)time(NULL);
	if( (now - st.trickle) > 86400 ){	// Once per day
		st.trickle = now;
		double progress = boinc_get_fraction_done();
		double cpu;
		boinc_wu_cpu_time(cpu);
		APP_INIT_DATA init_data;
		boinc_get_init_data(init_data);
		double run = boinc_elapsed_time() + init_data.starting_elapsed_time;
		char msg[512];
		sprintf(msg, "<trickle_up>\n"
			    "   <progress>%lf</progress>\n"
			    "   <cputime>%lf</cputime>\n"
			    "   <runtime>%lf</runtime>\n"
			    "</trickle_up>\n",
			     progress, cpu, run  );
		char variety[64];
		sprintf(variety, "wilson_progress");
		boinc_send_trickle_up(variety, msg);
	}
}

FILE *my_fopen(const char *filename, const char *mode){
	char resolved_name[512];
	boinc_resolve_filename(filename,resolved_name,sizeof(resolved_name));
	return boinc_fopen(resolved_name,mode);
}

void cleanup(progData & pd){
	sclReleaseMemObject(pd.d_primecount);
	sclReleaseMemObject(pd.d_totalcount);
	sclReleaseMemObject(pd.d_primes);
	sclReleaseMemObject(pd.d_testprimedata);
	sclReleaseMemObject(pd.d_residues);
	sclReleaseMemObject(pd.d_grptotal);
	sclReleaseMemObject(pd.d_found);
	sclReleaseMemObject(pd.d_acu);
	for(int i=0; i<3; ++i){
		sclReleaseMemObject(pd.d_powers[i]);
	}
	sclReleaseClSoft(pd.clearn);
	sclReleaseClSoft(pd.clearresult);
        sclReleaseClSoft(pd.iterate);
        sclReleaseClSoft(pd.setup);
        sclReleaseClSoft(pd.getsegprps);
        sclReleaseClSoft(pd.mulsmall);
        sclReleaseClSoft(pd.mullarge);
        sclReleaseClSoft(pd.reduce);
        sclReleaseClSoft(pd.finda);
        sclReleaseClSoft(pd.findc);
        sclReleaseClSoft(pd.findu);
        sclReleaseClSoft(pd.clearacu);
}

void write_state( searchData & sd, workStatus & st, cl_ulong2 * residues ){

	FILE * out;
	int eflag=0;

	// generate checkpoint file checksum
	st.state_sum = st.pmin + st.pmax + st.currp + st.trickle + st.totalcount + st.tpcount + st.done;
	for(uint32_t i=0; i<st.tpcount; ++i){
		st.state_sum += residues[i].s0 + residues[i].s1;
	}

        if (sd.write_state_a_next){
		if ((out = my_fopen(STATE_FILENAME_A,"wb")) == NULL)
			fprintf(stderr,"Cannot open %s !!!\n",STATE_FILENAME_A);
	}
	else{
                if ((out = my_fopen(STATE_FILENAME_B,"wb")) == NULL)
                        fprintf(stderr,"Cannot open %s !!!\n",STATE_FILENAME_B);
        }

	if(out != NULL){

		if( fwrite(&st, sizeof(workStatus), 1, out) != 1 ){
			fprintf(stderr,"Cannot write checkpoint to file. Continuing...\n");
			eflag = 1;
		}

		if( fwrite(residues, sizeof(cl_ulong2), st.tpcount, out) != st.tpcount ){
			fprintf(stderr,"Cannot write checkpoint to file. Continuing...\n");
			eflag = 1;
		}

		if(eflag){
			// Attempt to close, even though we failed to write
			fclose(out);
		}
		else{
			// If state file is closed OK, write to the other state file
			// next time around
			if (fclose(out) == 0) 
				sd.write_state_a_next = !sd.write_state_a_next; 
		}
	}
}

/* Return 1 only if a valid checkpoint can be read.
   Attempts to read from both state files,
   uses the most recent one available.
 */
int read_state( searchData & sd, workStatus & st, cl_ulong2 * residues ){

	FILE * in;
	bool good_state_a = true;
	bool good_state_b = true;
	workStatus stat_a, stat_b;
	cl_ulong2 *res_a, *res_b;

	res_a = (cl_ulong2 *)malloc(st.tpcount * sizeof(cl_ulong2));
	if( res_a == NULL ){
		fprintf(stderr,"malloc error, res_a array\n");
		exit(EXIT_FAILURE);
	}
	res_b = (cl_ulong2 *)malloc(st.tpcount * sizeof(cl_ulong2));
	if( res_b == NULL ){
		fprintf(stderr,"malloc error, res_b array\n");
		exit(EXIT_FAILURE);
	}

        // Attempt to read state file A
	if ((in = my_fopen(STATE_FILENAME_A,"rb")) == NULL){
		good_state_a = false;
        }
	else{
		if( fread(&stat_a, sizeof(workStatus), 1, in) != 1 ){
			fprintf(stderr,"Cannot parse %s !!!\n",STATE_FILENAME_A);
			printf("Cannot parse %s !!!\n",STATE_FILENAME_A);
			good_state_a = false;
		}
		else if( fread(res_a, sizeof(cl_ulong2), st.tpcount, in) != st.tpcount ){
			fprintf(stderr,"Cannot parse %s !!!\n",STATE_FILENAME_A);
			printf("Cannot parse %s !!!\n",STATE_FILENAME_A);
			good_state_a = false;
		}
		else if(stat_a.tpcount != st.tpcount || stat_a.pmin != st.pmin || stat_a.pmax != st.pmax){
			fprintf(stderr,"Invalid checkpoint file %s !!!\n",STATE_FILENAME_A);
			printf("Invalid checkpoint file %s !!!\n",STATE_FILENAME_A);
			good_state_a = false;
		}
		else if(stat_a.done){
			return 2;
		}
		else{
			uint64_t checksum = stat_a.pmin + stat_a.pmax + stat_a.currp + stat_a.trickle + stat_a.totalcount + stat_a.tpcount + stat_a.done;
			for(uint32_t i=0; i<st.tpcount; ++i){
				checksum += res_a[i].s0 + res_a[i].s1;
			}
			if(checksum != stat_a.state_sum){
				fprintf(stderr,"Checksum error in %s !!!\n",STATE_FILENAME_A);
				printf("Checksum error in %s !!!\n",STATE_FILENAME_A);
				good_state_a = false;
			}
		}
		fclose(in);
	}

        // Attempt to read state file B
	if ((in = my_fopen(STATE_FILENAME_B,"rb")) == NULL){
		good_state_b = false;
        }
	else{
		if( fread(&stat_b, sizeof(workStatus), 1, in) != 1 ){
			fprintf(stderr,"Cannot parse %s !!!\n",STATE_FILENAME_B);
			printf("Cannot parse %s !!!\n",STATE_FILENAME_B);
			good_state_b = false;
		}
		else if( fread(res_b, sizeof(cl_ulong2), st.tpcount, in) != st.tpcount ){
			fprintf(stderr,"Cannot parse %s !!!\n",STATE_FILENAME_B);
			printf("Cannot parse %s !!!\n",STATE_FILENAME_B);
			good_state_b = false;
		}
		else if(stat_b.tpcount != st.tpcount || stat_b.pmin != st.pmin || stat_b.pmax != st.pmax){
			fprintf(stderr,"Invalid checkpoint file %s !!!\n",STATE_FILENAME_B);
			printf("Invalid checkpoint file %s !!!\n",STATE_FILENAME_B);
			good_state_b = false;
		}
		else if(stat_b.done){
			return 2;
		}
		else{
			uint64_t checksum = stat_b.pmin + stat_b.pmax + stat_b.currp + stat_b.trickle + stat_b.totalcount + stat_b.tpcount + stat_b.done;
			for(uint32_t i=0; i<st.tpcount; ++i){
				checksum += res_b[i].s0 + res_b[i].s1;
			}
			if(checksum != stat_b.state_sum){
				fprintf(stderr,"Checksum error in %s !!!\n",STATE_FILENAME_B);
				printf("Checksum error in %s !!!\n",STATE_FILENAME_B);
				good_state_b = false;
			}
		}
		fclose(in);
	}

        // If both state files are OK, check which is the most recent
	if (good_state_a && good_state_b)
	{
		if (stat_a.currp > stat_b.currp)
			good_state_b = false;
		else
			good_state_a = false;
	}

	bool resume = false;

        // Use data from the most recent state file
	if (good_state_a && !good_state_b)
	{
		memcpy(residues, res_a, st.tpcount*sizeof(cl_ulong2));
		memcpy(&st, &stat_a, sizeof(workStatus));
		sd.write_state_a_next = false;
		resume = true;
	}
        if (good_state_b && !good_state_a)
        {
		memcpy(residues, res_b, st.tpcount*sizeof(cl_ulong2));
		memcpy(&st, &stat_b, sizeof(workStatus));
		sd.write_state_a_next = true;
		resume = true;
        }

	free(res_a);
	free(res_b);

	if(resume){
		return 1;
	}

	// If we got here, neither state file was good
	return 0;
}


void checkpoint(searchData & sd, workStatus & st, cl_ulong2 * residues, int checkpointTime){
	handle_trickle_up( st );
	write_state( sd, st, residues );
	boinc_checkpoint_completed();
	// display estimated time left if running standalone
	if(boinc_is_standalone() && checkpointTime && !sd.test){
		if(sd.lastp > 0.0){
			double progress = 100.0 * boinc_get_fraction_done();
			double diff = progress - sd.lastp;
			double left = 100.0 - progress;
			double psec = diff / (double)checkpointTime;
			uint64_t rem_sec = (uint64_t)( left / psec );
			uint64_t rem_days = rem_sec / 86400;
			rem_sec %= 86400;
			uint64_t rem_hours = rem_sec / 3600;
			rem_sec %= 3600;
			uint64_t rem_min = rem_sec / 60;
			rem_sec %= 60;
			sd.lastp = progress;

			printf("\rCheckpoint, Current P: %" PRIu64 ", eta: %" PRIu64 "d %" PRIu64 "h %" PRIu64 "m %" PRIu64 "s\n",
				st.currp, rem_days, rem_hours, rem_min, rem_sec);
		}
		else{
			sd.lastp = 100.0 * boinc_get_fraction_done();
			printf("\rCheckpoint, Current P: %" PRIu64 "\n", st.currp);
		}
	}
}


void getDataFromGPU( progData & pd, searchData & sd, sclHard hardware, workStatus & st, cl_ulong2 *residues, uint32_t * h_primecount ){

	uint64_t h_totalcount;

	// get residues from gpu (non-blocking)
	sclReadNB(hardware, st.tpcount * sizeof(cl_ulong2), pd.d_residues, residues);

	// copy prime count to host memory (non-blocking)
	sclReadNB(hardware, 3*sizeof(uint32_t), pd.d_primecount, h_primecount);

	// copy total prime count to host memory (blocking)
	sclRead(hardware, sizeof(uint64_t), pd.d_totalcount, &h_totalcount);

	// largest kernel prime count.  used to check array bounds
	if(h_primecount[1] > sd.psize){
		fprintf(stderr,"error: gpu prime array overflow\n");
		printf("error: gpu prime array overflow\n");
		exit(EXIT_FAILURE);
	}

	// flag set if there is a gpu overflow error
	if(h_primecount[2] == 1){
		fprintf(stderr,"error: getsegprps kernel local memory overflow\n");
		printf("error: getsegprps kernel local memory overflow\n");
		exit(EXIT_FAILURE);
	}

	// add total primes generated
	st.totalcount += h_totalcount;

}


// sleep CPU thread while waiting on the specified event to complete in the command queue
// using critical sections to prevent BOINC from shutting down the program while kernels are running on the GPU
void waitOnEvent(sclHard hardware, cl_event event){

	cl_int err;
	cl_int info;
#ifdef _WIN32
#else
	struct timespec sleep_time;
	sleep_time.tv_sec = 0;
	sleep_time.tv_nsec = 1000000;	// 1ms
#endif

	boinc_begin_critical_section();

	err = clFlush(hardware.queue);
	if ( err != CL_SUCCESS ) {
		printf( "ERROR: clFlush\n" );
		fprintf(stderr, "ERROR: clFlush\n" );
		sclPrintErrorFlags( err );
       	}

	while(true){

#ifdef _WIN32
		Sleep(1);
#else
		nanosleep(&sleep_time,NULL);
#endif

		err = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &info, NULL);
		if ( err != CL_SUCCESS ) {
			printf( "ERROR: clGetEventInfo\n" );
			fprintf(stderr, "ERROR: clGetEventInfo\n" );
			sclPrintErrorFlags( err );
	       	}

		if(info == CL_COMPLETE){
			err = clReleaseEvent(event);
			if ( err != CL_SUCCESS ) {
				printf( "ERROR: clReleaseEvent\n" );
				fprintf(stderr, "ERROR: clReleaseEvent\n" );
				sclPrintErrorFlags( err );
		       	}

			boinc_end_critical_section();

			return;
		}
	}
}


// queue a marker and sleep CPU thread until marker has been reached in the command queue
void sleepCPU(sclHard hardware){

	cl_event kernelsDone;
	cl_int err;
	cl_int info;
#ifdef _WIN32
#else
	struct timespec sleep_time;
	sleep_time.tv_sec = 0;
	sleep_time.tv_nsec = 1000000;	// 1ms
#endif

	boinc_begin_critical_section();

	// OpenCL v2.0
/*
	err = clEnqueueMarkerWithWaitList( hardware.queue, 0, NULL, &kernelsDone);
	if ( err != CL_SUCCESS ) {
		printf( "ERROR: clEnqueueMarkerWithWaitList\n");
		fprintf(stderr, "ERROR: clEnqueueMarkerWithWaitList\n");
		sclPrintErrorFlags(err); 
	}
*/
	err = clEnqueueMarker( hardware.queue, &kernelsDone);
	if ( err != CL_SUCCESS ) {
		printf( "ERROR: clEnqueueMarker\n");
		fprintf(stderr, "ERROR: clEnqueueMarker\n");
		sclPrintErrorFlags(err); 
	}

	err = clFlush(hardware.queue);
	if ( err != CL_SUCCESS ) {
		printf( "ERROR: clFlush\n" );
		fprintf(stderr, "ERROR: clFlush\n" );
		sclPrintErrorFlags( err );
       	}

	while(true){

#ifdef _WIN32
		Sleep(1);
#else
		nanosleep(&sleep_time,NULL);
#endif

		err = clGetEventInfo(kernelsDone, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &info, NULL);
		if ( err != CL_SUCCESS ) {
			printf( "ERROR: clGetEventInfo\n" );
			fprintf(stderr, "ERROR: clGetEventInfo\n" );
			sclPrintErrorFlags( err );
	       	}

		if(info == CL_COMPLETE){
			err = clReleaseEvent(kernelsDone);
			if ( err != CL_SUCCESS ) {
				printf( "ERROR: clReleaseEvent\n" );
				fprintf(stderr, "ERROR: clReleaseEvent\n" );
				sclPrintErrorFlags( err );
		       	}

			boinc_end_critical_section();

			return;
		}
	}
}



// find mod 30 wheel index based on starting N
// this is used by gpu threads to iterate over the number line
void findWheelOffset(uint64_t & start, int32_t & index){

	int32_t wheel[8] = {4, 2, 4, 2, 4, 6, 2, 6};
	int32_t idx = -1;

	// find starting number using mod 6 wheel
	// N=(k*6)-1, N=(k*6)+1 ...
	// where k, k+1, k+2 ...
	uint64_t k = start / 6;
	int32_t i = 1;
	uint64_t N = (k * 6)-1;


	while( N < start || N % 5 == 0 ){
		if(i){
			i = 0;
			N += 2;
		}
		else{
			i = 1;
			N += 4;
		}
	}

	start = N;

	// find mod 30 wheel index by iterating with a mod 6 wheel until finding N divisible by 5
	// forward to find index
	while(idx < 0){

		if(i){
			N += 2;
			i = 0;
			if(N % 5 == 0){
				N -= 2;
				idx = 5;
			}

		}
		else{
			N += 4;
			i = 1;
			if(N % 5 == 0){
				N -= 4;
				idx = 7;
			}
		}
	}

	// reverse to find starting index
	while(N != start){
		--idx;
		if(idx < 0)idx=7;
		N -= wheel[idx];
	}


	index = idx;

}


int64_t getACU( progData & pd, sclHard hardware, uint64_t p ){

	uint32_t * h_found = (uint32_t *)malloc(sizeof(uint32_t));
	if( h_found == NULL ){
		fprintf(stderr,"malloc error\n");
		exit(EXIT_FAILURE);
	}

	// copy result count to host memory
	// blocking read
	sclRead(hardware, sizeof(uint32_t), pd.d_found, h_found);

	if(!h_found[0]){
		printf("ERROR: acu not found for p: %" PRIu64 "!\n",p);
		fprintf(stderr,"ERROR: acu not found for p: %" PRIu64 "!\n",p);
		exit(EXIT_FAILURE);
	}

	// copy results to host memory
	// blocking read
	int64_t * h_acu = (int64_t *)malloc(h_found[0] * sizeof(int64_t));
	if( h_acu == NULL ){
		fprintf(stderr,"malloc error\n");
		exit(EXIT_FAILURE);
	}

	sclRead(hardware, h_found[0] * sizeof(cl_long), pd.d_acu, h_acu);

/*	printf("%u acus\n",h_found[0]);

	for(uint32_t j=0; j<h_found[0]; ++j){
		printf("acu result %" PRId64 "\n",h_acu[j]);
	}
*/
	int64_t acu = h_acu[0];

	free(h_acu);
	free(h_found);

	return acu;

}


// finds a as solution of a^2+b^2=p && a=1 (mod 4)
int64_t find_a(uint64_t p, progData & pd, sclHard hardware){

	uint64_t maxa = (uint64_t)sqrtl( (long double)p );

	sclEnqueueKernel(hardware, pd.clearacu);

	sclSetKernelArg(pd.finda, 2, sizeof(uint64_t), &p);
	sclSetKernelArg(pd.finda, 3, sizeof(uint64_t), &maxa);
	sclEnqueueKernel(hardware, pd.finda);

	int64_t a = getACU(pd, hardware, p);

	return a;
}

// finds c as solution of c^2+27d^2=4p && c=1 (mod 3)
int64_t find_c(uint64_t p, progData & pd, sclHard hardware){

	int64_t c=0;

	if(p <= maxp){
		uint64_t p4=4*p;
		uint64_t maxd = (uint64_t)sqrtl( ((long double)p4) / 27.0 );

		sclEnqueueKernel(hardware, pd.clearacu);

		sclSetKernelArg(pd.findc, 2, sizeof(uint64_t), &p4);
		sclSetKernelArg(pd.findc, 3, sizeof(uint64_t), &maxd);
		sclEnqueueKernel(hardware, pd.findc);

		c = getACU(pd, hardware, p);
	}
	else{
		printf("P: %" PRIu64 " is too large for find_c!\n", p);
		fprintf(stderr,"P: %" PRIu64 " is too large for find_c!\n", p);
		exit(EXIT_FAILURE);
	}

	return c;
}

// finds u as solution of u^2+3v^2=4p && u=1 (mod 3)
int64_t find_u(uint64_t p, progData & pd, sclHard hardware){

	int64_t u=0;

	if(p <= maxp){
		uint64_t p4=4*p;
		uint64_t maxv = (uint64_t)sqrtl( ((long double)p4) / 3.0 );
		uint32_t umod=(((p-1)/6)%2==0) ? 1 : 2;

		sclEnqueueKernel(hardware, pd.clearacu);

		sclSetKernelArg(pd.findu, 2, sizeof(uint64_t), &p4);
		sclSetKernelArg(pd.findu, 3, sizeof(uint64_t), &maxv);
		sclSetKernelArg(pd.findu, 4, sizeof(uint32_t), &umod);
		sclEnqueueKernel(hardware, pd.findu);

		u = getACU(pd, hardware, p);
	}
	else{
		printf("P: %" PRIu64 " is too large for find_u!\n", p);
		fprintf(stderr,"P: %" PRIu64 " is too large for find_u!\n", p);
		exit(EXIT_FAILURE);
	}

	return u;
}


void writeResult(uint64_t p, int32_t b){

	FILE *out;

	if ((out = my_fopen(RESULT_FILENAME,"a")) == NULL){
		fprintf(stderr,"Cannot open %s !!!\n",RESULT_FILENAME);
		printf("Cannot open %s !!!\n",RESULT_FILENAME);
		exit(EXIT_FAILURE);
	}

	if( b == 0 ){
		if( fprintf(out,"%" PRIu64 " is a Wilson prime\n", p) < 0 ){
			fprintf(stderr,"Cannot write to %s !!!\n",RESULT_FILENAME);
			printf("Cannot write to %s !!!\n",RESULT_FILENAME);
			exit(EXIT_FAILURE);
		}
	}
	else{
		if( fprintf(out,"%" PRIu64 " is a Near-Wilson prime %+d\n", p, b) < 0 ){
			fprintf(stderr,"Cannot write to %s !!!\n",RESULT_FILENAME);
			printf("Cannot write to %s !!!\n",RESULT_FILENAME);
			exit(EXIT_FAILURE);
		}
	}

	if(fclose(out) != 0){
		fprintf(stderr,"Cannot close %s !!!\n",RESULT_FILENAME);
		printf("Cannot close %s !!!\n",RESULT_FILENAME);
		exit(EXIT_FAILURE);
	}	

}


void processResult(uint64_t p, uint64_t s0, uint64_t s1, uint32_t type, progData & pd, searchData & sd, sclHard hardware, uint64_t * prps){

	mpz_t residue, psq, mp, a, b;
	
	mpz_init(mp);
	mpz_import(mp, 1, 1, sizeof(uint64_t), 0, 0, &p);
	
	mpz_init(a);
	mpz_init(b);
	mpz_import(a, 1, 1, sizeof(uint64_t), 0, 0, &s0);
	mpz_import(b, 1, 1, sizeof(uint64_t), 0, 0, &s1);
	mpz_init(residue);	
	mpz_mul(residue, mp, b);
	mpz_add(residue, residue, a);

	mpz_init(psq);
	mpz_mul(psq, mp, mp);

	// divide prps out of residue
	uint64_t i;
	for(i=0; i<PRPSIZE; ++i){
		uint64_t theprp = prps[i];
		if(theprp > sd.typeTarget[type]) break;
		mpz_import(a, 1, 1, sizeof(uint64_t), 0, 0, &theprp);
		int reti = mpz_invert(a, a, psq);
		if(!reti){
			printf("ERROR: inverse doesn’t exist, prp: %" PRIu64 " testprime: %" PRIu64 "\n", theprp, p);
			fprintf(stderr,"ERROR: inverse doesn’t exist, prp: %" PRIu64 " testprime: %" PRIu64 "\n", theprp, p);
			exit(EXIT_FAILURE);
		}
		uint32_t thepower = sd.typeTarget[type] / theprp;
		for(uint32_t j=0; j<thepower; ++j){
			mpz_mul(residue, residue, a);
			mpz_mod(residue, residue, psq);
		}
	}
	if(i > sd.prpsremoved){
		sd.prpsremoved = i;
	}

	if(type == 0){
		int64_t uu = find_u(p, pd, hardware);
		int64_t cc = find_c(p, pd, hardware);
		mpz_t mu, mc;
		mpz_init(mu);
		mpz_init(mc);		
		if(uu < 0){
			uu = -uu;
			mpz_import(mu, 1, 1, sizeof(int64_t), 0, 0, &uu);
			mpz_neg(mu, mu);
		}
		else{
			mpz_import(mu, 1, 1, sizeof(int64_t), 0, 0, &uu);
		}
		if(cc < 0){
			cc = -cc;
			mpz_import(mc, 1, 1, sizeof(int64_t), 0, 0, &cc);
			mpz_neg(mc, mc);
		}
		else{
			mpz_import(mc, 1, 1, sizeof(int64_t), 0, 0, &cc);
		}		
		// residue = ((p-1)/6)! (mod p^2)
		mpz_powm_ui(residue, residue, 6, psq);
		// residue = ((p-1)/6)!^6 (mod p^2)
		mpz_set_ui(a, 2);
		mpz_powm(a, a, mp, psq);
		mpz_sub_ui(a, a, 1);
		// a = (2^p)-1
		mpz_pow_ui(b, mu, 3);
		mpz_neg(b, b);
		// b = −u^3
		mpz_mul(a, b, a);
		// a = -u^3 * 2^p-1
		mpz_mul(b, mu, mp);
		mpz_mul_ui(b, b, 3);
		// b = 3*p*u
		mpz_add(a, a, b);
		// a = (-u^3 * (2^p-1)) + 3pu
		mpz_mul(residue, residue, a);
		int reti = mpz_invert(a, mc, psq);
		if(!reti){
			printf("ERROR: inverse doesn’t exist, c: %" PRId64 " testprime: %" PRIu64 "\n", cc, p);
			fprintf(stderr,"ERROR: inverse doesn’t exist, c: %" PRId64 " testprime: %" PRIu64 "\n", cc, p);
			exit(EXIT_FAILURE);
		}		
		mpz_mul(a, mp, a);
		mpz_sub(a, a, mc);
		// a = p/c-c
		mpz_mul(residue, residue, a);
		// residue = (((p-1)/6)!^6) * ((-u^3 * (2^p-1)) + 3*p*u) * (p/c-c)
		mpz_set_ui(a, 3);
		mpz_powm(a, a, mp, psq);
		mpz_sub_ui(a, a, 1);
		// a = 3^p-1
		mpz_set_ui(b, 2);
		reti = mpz_invert(b, b, psq);
		if(!reti){
			printf("ERROR: inverse doesn’t exist, val: 2 testprime: %" PRIu64 "\n", p);
			fprintf(stderr,"ERROR: inverse doesn’t exist, val: 2 testprime: %" PRIu64 "\n", p);
			exit(EXIT_FAILURE);
		}
		mpz_mul(a, a, b);
		// a = (3^p-1)/2
		mpz_mul(residue, residue, a);
		// residue = (((p-1)/6)!^6) * ((-u^3 * (2^p-1)) + 3*p*u) * (p/c-c) * ((3^p-1)/2)
		// which is congruent to (p-1)!  when p = 1 mod 3
		mpz_clear(mu);
		mpz_clear(mc);
	}
	else if(type == 1){
		int64_t aa = find_a(p, pd, hardware);
		mpz_t ma;
		mpz_init(ma);
		if(aa < 0){
			aa = -aa;
			mpz_import(ma, 1, 1, sizeof(int64_t), 0, 0, &aa);
			mpz_neg(ma, ma);
		}
		else{
			mpz_import(ma, 1, 1, sizeof(int64_t), 0, 0, &aa);
		}		
		// residue = ((p-1)/4)! (mod p^2)
		mpz_powm_ui(residue, residue, 4, psq);
		// residue = ((p-1)/4)!^4 (mod p^2)
		mpz_set_ui(a, 2);
		mpz_powm(a, a, mp, psq);
		// a = 2^p (mod p^2)
		mpz_mul_ui(a, a, 3);
		// a = 3*2^p
		mpz_sub_ui(a, a, 4);
		// a = 3*2^p-4
		mpz_mul(residue, residue, a);
		// residue = ((p-1)/4)!^4 * (3*2^p-4)
		mpz_mul(a, ma, ma);
		mpz_mul_ui(a, a, 2);
		mpz_sub(a, a, mp);
		// a = 2*a^2-p
		mpz_mul(residue, residue, a);
		// residue = ((p-1)/4)!^4 * (3*2^p-4) * (2*a^2-p)
		// which is congruent to (p-1)!  when p = 5 mod 12
		mpz_clear(ma);
	}
	else if(type == 2){
		// residue = ((p-1)/2)! (mod p^2)
		mpz_mul(residue, residue, residue);
		// residue = ((p-1)/2)!^2 (mod p^2)
		mpz_set_ui(a, 2);
		mpz_powm(a, a, mp, psq);
		// a = 2^p (mod p^2)
		mpz_ui_sub(a, 1, a);
		// a = 1-2^p
		mpz_mul(residue, residue, a);
		// residue = ((p-1)/2)!^2 * (1-2^p)
		// which is congruent to (p-1)! when p = 11 mod 12
	}

	// add 1 and final mod
	mpz_add_ui(residue, residue, 1);
	mpz_mod(residue, residue, psq);
	// res = (p-1)! + 1 (mod p^2)

	mpz_tdiv_qr(a, b, residue, mp);
	uint64_t rem=0, quot=0;
	mpz_export(&quot, NULL, 1, sizeof(uint64_t), 0, 0, a);
	mpz_export(&rem, NULL, 1, sizeof(uint64_t), 0, 0, b);
	
	mpz_clear(residue);
	mpz_clear(psq);
	mpz_clear(mp);
	mpz_clear(a);
	mpz_clear(b);

	// Verify our calculations were correct
	// From Wilson’s theorem it follows that the Wilson quotient is an integer only if p is not composite
	if(rem != 0){
		fprintf(stderr,"error: Wilson quotient check failed! p: %" PRIu64 " type: %u rem: %" PRIu64 "\n", p, type, rem);
		printf("error: Wilson quotient check failed! p: %" PRIu64 " type: %u rem: %" PRIu64 "\n", p, type, rem);
		exit(EXIT_FAILURE);
	}

	uint64_t negquot = p-quot;
	uint64_t smallest = (quot <= negquot) ? quot : negquot;
	
/*	// |w_p/p| < 1/50000 for testing with known good residue file
	// see A SEARCH FOR WILSON PRIMES EDGAR COSTA, ROBERT GERBICZ, AND DAVID HARVEY
	const double SPECIAL_THRESHOLD = 1.0/50000.0;
	double wpp = (double)((long double)smallest / (long double)p);
*/
	const uint64_t SPECIAL_THRESHOLD = 1000;

	if( quot == 0 ){
		if(boinc_is_standalone()){
			printf("%" PRIu64 " is a Wilson prime\n", p);
		}
		writeResult(p, 0);
		++sd.resultcount;
		if(sd.test){
			sd.testResultPrime = p;
			sd.testResultValue = 0;
		}
	}
//	else if( wpp < SPECIAL_THRESHOLD ){
	else if( smallest < SPECIAL_THRESHOLD ){
		int32_t dq = (smallest == quot) ? quot : -((int32_t)negquot);
		if(boinc_is_standalone()){
			printf("%" PRIu64 " is a Near-Wilson prime %+d\n", p, dq);
		}
		writeResult(p, dq);
		++sd.resultcount;
		if(sd.test){
			sd.testResultPrime = p;
			sd.testResultValue = dq;		
		}
	}

	sd.checksum += p + rem + quot;

}


void getResults(progData & pd, searchData & sd, sclHard hardware, workStatus & st, cl_ulong2 *residues, testPrime *tp){

	if(boinc_is_standalone()){
		printf("Finalizing results on cpu\n");
	}
	
	// read file of 2-PRPs
	FILE *in;
	in = my_fopen("prps.dat", "rb");
	if(in == NULL) {
		fprintf(stderr,"error opening prp file\n");
		printf("error opening prp file\n");
		exit(EXIT_FAILURE);
	}
	fseek(in, 0, SEEK_END);
	uint64_t file_size = ftell(in);
	rewind(in);
	uint64_t prpcount = file_size / sizeof(uint64_t);
	if(prpcount != PRPSIZE){
		fprintf(stderr,"prp file read error, file size is incorrect\n");
		printf("prp file read error, file size is incorrect\n");
		exit(EXIT_FAILURE);
	}
	uint64_t * prps = (uint64_t *)malloc(PRPSIZE*sizeof(uint64_t));
	if(prps == NULL){
		fprintf(stderr,"malloc error, prps array\n");
		printf("malloc error, prps array\n");
		exit(EXIT_FAILURE);
	}
	size_t read = fread(prps, sizeof(uint64_t), PRPSIZE, in);
	if(read != PRPSIZE) {
		fprintf(stderr,"prp file read error\n");
		printf("prp file read error\n");
		free(prps);
		fclose(in);
		exit(EXIT_FAILURE);
	}
	fclose(in);
	uint64_t prpsum = 0;
	for(uint32_t i=0; i<PRPSIZE; ++i){
		prpsum += prps[i];
	}
	if(prpsum != 0x959601167DFEE126){
		fprintf(stderr,"prp file checksum error\n");
		printf("prp file checksum error\n");
		exit(EXIT_FAILURE);
	}
	// finalize each prime's result
	for(uint32_t j=0; j<st.tpcount; ++j){
		processResult(tp[j].p, residues[j].s0, residues[j].s1, tp[j].type, pd, sd, hardware, prps);
	}
	free(prps);
}


void finalizeResults(searchData & sd){

	char line[256];
	uint32_t lc = 0;
	FILE * resfile;

	if(sd.resultcount){
		// check result file has the same number of lines as the result count
		resfile = my_fopen(RESULT_FILENAME,"r");

		if(resfile == NULL){
			fprintf(stderr,"Cannot open %s !!!\n",RESULT_FILENAME);
			exit(EXIT_FAILURE);
		}

		while(fgets(line, sizeof(line), resfile) != NULL) {
			++lc;
		}

		fclose(resfile);

		if(lc < sd.resultcount){
			fprintf(stderr,"ERROR: Missing results in %s !!!\n",RESULT_FILENAME);
			printf("ERROR: Missing results in %s !!!\n",RESULT_FILENAME);
			exit(EXIT_FAILURE);
		}
	}

	resfile = my_fopen(RESULT_FILENAME,"a");
	if( resfile == NULL ){
		fprintf(stderr,"Cannot open %s !!!\n",RESULT_FILENAME);
		exit(EXIT_FAILURE);
	}
	if(sd.resultcount == 0){
		if( fprintf( resfile, "no results\n%016" PRIX64 "\n", sd.checksum ) < 0 ){
			fprintf(stderr,"Cannot write to %s !!!\n",RESULT_FILENAME);
			exit(EXIT_FAILURE);
		}
	}
	else{
		if( fprintf( resfile, "%016" PRIX64 "\n", sd.checksum ) < 0 ){
			fprintf(stderr,"Cannot write to %s !!!\n",RESULT_FILENAME);
			exit(EXIT_FAILURE);
		}
	}
	if(fclose(resfile) != 0){
		fprintf(stderr,"Cannot close %s !!!\n",RESULT_FILENAME);
		printf("Cannot close %s !!!\n",RESULT_FILENAME);
		exit(EXIT_FAILURE);
	}
}


void setupSearch(workStatus & st){

	st.currp = 2;

	if(st.pmin == 0 || st.pmax == 0){
		printf("-p and -P arguments are required\nuse -h for help\n");
		fprintf(stderr, "-p and -P arguments are required\n");
		exit(EXIT_FAILURE);
	}

	if(st.pmin > st.pmax){
		printf("pmin <= pmax is required\nuse -h for help\n");
		fprintf(stderr, "pmin <= pmax is required\n");
		exit(EXIT_FAILURE);
	}
	
	if(st.pmax > st.pmin + 10000000){
		printf("range <= 10000000 is required\nuse -h for help\n");
		fprintf(stderr, "range <= 10000000 is required\n");
		exit(EXIT_FAILURE);
	}	
	
	fprintf(stderr, "Starting search at p: %" PRIu64 "\nStopping search at P: %" PRIu64 "\n", st.pmin, st.pmax);
	if(boinc_is_standalone()){
		printf("Starting search at p: %" PRIu64 "\nStopping search at P: %" PRIu64 "\n", st.pmin, st.pmax);
	}

}


void profileGPU(progData & pd, searchData & sd, sclHard hardware){

	// calculate approximate chunk size based on gpu's compute units
	cl_int err = 0;
	
	uint64_t calc_range = sd.computeunits * (uint64_t)1510000;

	// limit kernel global size
	if(calc_range > 4294900000){
		calc_range = 4294900000;
	}
	
	uint64_t start = 0xFFFFFFFF;
	uint64_t stop = start + calc_range;

	sclSetGlobalSize( pd.getsegprps, calc_range/60+1 );

	// get a count of primes in the gpu worksize
	uint64_t range_primes = (stop / log(stop)) - (start / log(start));

	// calculate prime array size based on result
	uint64_t mem_size = (uint64_t)(1.5 * (double)range_primes);

	// kernels use uint for global id
	if(mem_size > UINT32_MAX){
		fprintf(stderr, "ERROR: mem_size too large.\n");
                printf( "ERROR: mem_size too large.\n" );
		exit(EXIT_FAILURE);
	}

	pd.d_primes = clCreateBuffer(hardware.context, CL_MEM_READ_WRITE, mem_size*sizeof(cl_ulong), NULL, &err);
        if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure d_primes\n");
                printf( "ERROR: clCreateBuffer failure d_primes\n" );
		exit(EXIT_FAILURE);
	}
	for(int i=0; i<3; ++i){
		pd.d_powers[i] = clCreateBuffer(hardware.context, CL_MEM_READ_WRITE, mem_size*sizeof(cl_uint2), NULL, &err);
		if ( err != CL_SUCCESS ) {
			fprintf(stderr, "ERROR: clCreateBuffer failure d_powers\n");
			printf( "ERROR: clCreateBuffer failure d_powers\n" );
			exit(EXIT_FAILURE);
		}
	}

	int32_t wheelidx;
	uint64_t kernel_start = start;
	findWheelOffset(kernel_start, wheelidx);

	sclSetKernelArg(pd.getsegprps, 0, sizeof(uint64_t), &kernel_start);
	sclSetKernelArg(pd.getsegprps, 1, sizeof(uint64_t), &stop);
	sclSetKernelArg(pd.getsegprps, 2, sizeof(int32_t), &wheelidx);
	sclSetKernelArg(pd.getsegprps, 3, sizeof(cl_mem), &pd.d_primes);
	sclSetKernelArg(pd.getsegprps, 4, sizeof(cl_mem), &pd.d_primecount);
	sclSetKernelArg(pd.getsegprps, 5, sizeof(cl_mem), &pd.d_powers[0]);
	sclSetKernelArg(pd.getsegprps, 6, sizeof(cl_mem), &pd.d_powers[1]);
	sclSetKernelArg(pd.getsegprps, 7, sizeof(cl_mem), &pd.d_powers[2]);
	sclSetKernelArg(pd.getsegprps, 8, sizeof(uint64_t), &sd.typeTarget[0]);
	sclSetKernelArg(pd.getsegprps, 9, sizeof(uint64_t), &sd.typeTarget[1]);
	sclSetKernelArg(pd.getsegprps, 10, sizeof(uint64_t), &sd.typeTarget[2]);
	sclSetKernelArg(pd.getsegprps, 11, sizeof(uint64_t), &sd.powerLimit[0]);
	sclSetKernelArg(pd.getsegprps, 12, sizeof(uint64_t), &sd.powerLimit[1]);
	sclSetKernelArg(pd.getsegprps, 13, sizeof(uint64_t), &sd.powerLimit[2]);

	// zero prime count
	sclEnqueueKernel(hardware, pd.clearresult);

	// Benchmark the GPU
	double kernel_ms = ProfilesclEnqueueKernel(hardware, pd.getsegprps);

	// target runtime for prime generator kernel is approx 3.0 ms
	double prof_multi = 3.0 / kernel_ms;

	// update chunk size based on the profile
	calc_range = (uint64_t)( (double)calc_range * prof_multi );

	// limit kernel global size
	if(calc_range > 4294900000){
		calc_range = 4294900000;
	}

	// get a count of primes in the new gpu worksize
	stop = start + calc_range;

	range_primes = (stop / log(stop)) - (start / log(start));

	// calculate prime array size based on result
	mem_size = (uint64_t)( 1.5 * (double)range_primes );

	if(mem_size > UINT32_MAX){
		fprintf(stderr, "ERROR: mem_size too large.\n");
                printf( "ERROR: mem_size too large.\n" );
		exit(EXIT_FAILURE);
	}

	sd.range = calc_range;
	sd.psize = mem_size;
	
	fprintf(stderr, "r:%u p:%u\n",sd.range,sd.psize);

	// free temporary arrays
	sclReleaseMemObject(pd.d_primes);
	for(int i=0; i<3; ++i){
		sclReleaseMemObject(pd.d_powers[i]);
	}

}


cl_ulong2 getPower(uint32_t prime, uint64_t target){

	if(prime > target){
		return (cl_ulong2){0,0};
	}
	uint64_t totalpower = 0;
	uint64_t currp = prime;
	uint64_t q = target / currp;
	while(q){
		totalpower += q;
		unsigned __int128 pp = (unsigned __int128)currp * prime;
		if(pp > target) break;
		currp = pp;
		q = target / currp;
	}
	uint64_t curBit = 0x8000000000000000;
	if(totalpower > 1){
		curBit >>= ( __builtin_clzll(totalpower) + 1 );
	}
	return (cl_ulong2){totalpower, curBit};
}


void get32bitprimes(sclHard hardware, progData & pd, searchData & sd, workStatus & st, uint64_t * smprime,
			cl_ulong2 * smpower, cl_ulong * h_prime, cl_ulong2 * h_power, primesieve_iterator & it, uint64_t stop){

	// get a segment of primes
	uint32_t smcount = 0;
	for(; smcount < sd.psize; ++smcount){
		uint64_t prime = primesieve_next_prime(&it);
		if(prime >= stop){
			prime = primesieve_prev_prime(&it);
			break;
		}
		smprime[smcount] = prime;
	}

	// generate compressed prime and power tables for all 3 prime types
	for(uint32_t t=0; t<3; ++t){
		if(!sd.tpcnt[t] || smprime[0] > sd.typeTarget[t]){
			sd.pcount32[t] = 0;
			continue;
		}
		uint32_t newcount = smcount;
		for(uint32_t b = 0; b < smcount; ++b){
			if(smprime[b] > sd.typeTarget[t]){
				newcount = b;
				break;
			}
			smpower[b] = getPower(smprime[b], sd.typeTarget[t]);
		}
		// compress the power table by combining primes with the same power
		// skip the first prime, therefore, the power table will have at least one term
		h_prime[0] = smprime[0];
		h_power[0] = smpower[0];
		uint32_t m=1;
		for(uint32_t i=1; i<newcount; ++m){
			h_prime[m] = smprime[i];
			h_power[m] = smpower[i];
			for(++i; i<newcount && h_power[m].s0 == smpower[i].s0; ++i){
				unsigned __int128 pp = (unsigned __int128)h_prime[m] * smprime[i];
				if(pp > 0xFFFFFFFFFFFFFFFF) break;
				h_prime[m] = pp;
			}
		}
		sclWriteNB(hardware, m * sizeof(cl_ulong), pd.d_primes32[t], h_prime);
		sclWrite(hardware, m * sizeof(cl_ulong2), pd.d_powers32[t], h_power);
		sd.pcount32[t] = m;
	}
	
	// add total primes generated
	st.totalcount += smcount;	
}


uint64_t getPrimes(sclHard hardware, progData & pd, searchData & sd, workStatus & st, uint64_t * smprime,
			cl_ulong2 * smpower, cl_ulong * h_prime, cl_ulong2 * h_power, primesieve_iterator & it){

	uint64_t stop = st.currp + sd.range;
	if(stop > sd.maxtarget+1){
		stop = sd.maxtarget+1;
	}
	
	if(st.currp < 0xFFFFFFFF && stop > 0xFFFFFFFF){
		stop = 0xFFFFFFFF;
	}
	
	if(st.currp < 0xFFFFFFFF){
		get32bitprimes(hardware, pd, sd, st, smprime, smpower, h_prime, h_power, it, stop);
	}
	else{
		int32_t wheelidx;
		uint64_t kernel_start = st.currp;
		findWheelOffset(kernel_start, wheelidx);
		sclSetKernelArg(pd.getsegprps, 0, sizeof(uint64_t), &kernel_start);
		sclSetKernelArg(pd.getsegprps, 1, sizeof(uint64_t), &stop);
		sclSetKernelArg(pd.getsegprps, 2, sizeof(int32_t), &wheelidx);
		sclEnqueueKernel(hardware, pd.getsegprps);
//		float kernel_ms = ProfilesclEnqueueKernel(hardware, pd.getsegprps);
//		printf("getsegprps %0.2fms\n",kernel_ms);
	}

	return stop;

}


void multiply(sclHard hardware, progData & pd, searchData & sd, workStatus & st, uint32_t tpnum, uint32_t type){

	if(st.currp < 0xFFFFFFFF){
		sclSetKernelArg(pd.mulsmall, 1, sizeof(cl_mem), &pd.d_primes32[type]);
		sclSetKernelArg(pd.mulsmall, 2, sizeof(cl_mem), &pd.d_powers32[type]);
		sclSetKernelArg(pd.mulsmall, 4, sizeof(uint32_t), &tpnum);
		sclSetKernelArg(pd.mulsmall, 5, sizeof(uint32_t), &sd.pcount32[type]);
		sclEnqueueKernel(hardware, pd.mulsmall);
//		float kernel_ms = ProfilesclEnqueueKernel(hardware, pd.mulsmall);
//		printf("mulsmall %0.2fms\n",kernel_ms);
	}
	else{
		sclSetKernelArg(pd.mullarge, 3, sizeof(cl_mem), &pd.d_powers[type]);
		sclSetKernelArg(pd.mullarge, 5, sizeof(uint32_t), &tpnum);
		sclSetKernelArg(pd.mullarge, 6, sizeof(uint64_t), &sd.powerLimit[type]);
		sclSetKernelArg(pd.mullarge, 7, sizeof(uint64_t), &sd.typeTarget[type]);
		sclEnqueueKernel(hardware, pd.mullarge);
//		float kernel_ms = ProfilesclEnqueueKernel(hardware, pd.mullarge);
//		printf("mullarge %0.2fms\n",kernel_ms);
	}

}


void getFractionDone(searchData & sd, workStatus & st, double partial){

	// simplified fraction done.  fraction done will speed up as workunit progresses.
	// a more accurate calculation can be done by counting power table primes per test p.
	// however, in practice this requires too many calcuations!
	long double currTotal = (long double)st.currp + partial;

	double fd = (double)(currTotal / (long double)sd.maxtarget);
	
	if(fd > 1.0) fd = 1.0;

	if(fd < 1.0){
		boinc_fraction_done(fd);
		if(boinc_is_standalone()){
			printf("%.4f%%\n",fd*100.0);
		}
	}
}


void cl_sieve( sclHard hardware, searchData & sd, workStatus & st ){

	progData pd = {};
	testPrime *tp;
	time_t boinc_last, ckpt_last, time_curr;
	cl_int err = 0;

	// setup kernel parameters
	setupSearch(st);

	// arrays used to transfer data from gpu during checkpoints
	cl_ulong2 *residues;
	uint32_t * h_primecount = (uint32_t *)malloc(3*sizeof(uint32_t));
	if( h_primecount == NULL ){
		fprintf(stderr,"malloc error, h_primecount\n");
		exit(EXIT_FAILURE);
	}
	pd.d_primecount = clCreateBuffer( hardware.context, CL_MEM_READ_WRITE, 3*sizeof(cl_uint), NULL, &err );
        if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure.\n");
                printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}
	pd.d_totalcount = clCreateBuffer( hardware.context, CL_MEM_READ_WRITE, sizeof(cl_ulong), NULL, &err );
        if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure.\n");
                printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}

	// build kernels
        pd.setup = sclGetCLSoftwareWithCommon(common_cl, setup_cl,"setup",hardware,NULL);
        pd.iterate = sclGetCLSoftwareWithCommon(common_cl, iterate_cl,"iterate",hardware,NULL);
        pd.mulsmall = sclGetCLSoftwareWithCommon(common_cl, mulsmall_cl,"mulsmall",hardware,NULL);
        pd.mullarge = sclGetCLSoftwareWithCommon(common_cl, mullarge_cl,"mullarge",hardware,NULL);
        pd.reduce = sclGetCLSoftwareWithCommon(common_cl, reduce_cl,"reduce",hardware,NULL);

        pd.clearn = sclGetCLSoftware(clearn_cl,"clearn",hardware,NULL);
        pd.clearresult = sclGetCLSoftware(clearresult_cl,"clearresult",hardware,NULL);        
        pd.getsegprps = sclGetCLSoftware(getsegprps_cl,"getsegprps",hardware,NULL);
        pd.finda = sclGetCLSoftware(find_cl,"finda",hardware,NULL);
        pd.findc = sclGetCLSoftware(find_cl,"findc",hardware,NULL);
        pd.findu = sclGetCLSoftware(find_cl,"findu",hardware,NULL);
        pd.clearacu = sclGetCLSoftware(find_cl,"clearacu",hardware,NULL);

	// kernels have __attribute__ ((reqd_work_group_size(256, 1, 1)))
	// it's still possible the CL complier picked a different size
	if(pd.getsegprps.local_size[0] != 256){
		pd.getsegprps.local_size[0] = 256;
		fprintf(stderr, "Set getsegprps kernel local size to 256\n");
	}
	if(pd.mulsmall.local_size[0] != 256){
		pd.mulsmall.local_size[0] = 256;
		fprintf(stderr, "Set mulsmall kernel local size to 256\n");
	}
	if(pd.mullarge.local_size[0] != 256){
		pd.mullarge.local_size[0] = 256;
		fprintf(stderr, "Set mullarge kernel local size to 256\n");
	}
	// local size is 1024 for nvidia, 256 for all others
	if(sd.nvidia){
		if(pd.reduce.local_size[0] != 1024){		// cl compiler picks 256!
			pd.reduce.local_size[0] = 1024;
		}
		sclSetGlobalSize( pd.reduce, 1024 );
		
		if(pd.iterate.local_size[0] != 1024){		// cl compiler picks 256!
			pd.iterate.local_size[0] = 1024;
		}
		sclSetGlobalSize( pd.iterate, 1024 );		
	}
	else{
		if(pd.reduce.local_size[0] != 256){
			pd.reduce.local_size[0] = 256;
			fprintf(stderr, "Set reduce kernel local size to 256\n");
		}
		sclSetGlobalSize( pd.reduce, 256 );
		
		if(pd.iterate.local_size[0] != 256){
			pd.iterate.local_size[0] = 256;
			fprintf(stderr, "Set iterate kernel local size to 256\n");
		}
		sclSetGlobalSize( pd.iterate, 256 );				
	}	

	// setup primes to test
	size_t tpsize;
	uint64_t *tplist = (uint64_t*)primesieve_generate_primes(st.pmin, st.pmax-1, &tpsize, UINT64_PRIMES);
	st.tpcount = (uint32_t)tpsize;

	if( !st.tpcount ){
		fprintf(stderr, "there are no primes to test in this range!\n");
		printf( "there are no primes to test in this range!\n");
		exit(EXIT_FAILURE);
	}

	tp = (testPrime *)malloc(st.tpcount * sizeof(testPrime));
	if( tp == NULL ){
		fprintf(stderr,"malloc error, testPrime array\n");
		exit(EXIT_FAILURE);
	}
	residues = (cl_ulong2 *)malloc(st.tpcount * sizeof(cl_ulong2));
	if( residues == NULL ){
		fprintf(stderr,"malloc error, residue array\n");
		exit(EXIT_FAILURE);
	}

	// our target factorial is ((p-1)/n)! using the first prime of the type in the test range
	// the remaining test primes will have iterations added to this target
	// powerLimit is the transition point where the power of the primes used to calculate the factorial target is 1
	for(uint32_t i=0; i<st.tpcount; ++i){
		tp[i].p = tplist[i];
		if(tplist[i] % 3 == 1){
			++sd.tpcnt[0];
			tp[i].type = 0;
			tp[i].pTarget = (tplist[i]-1)/6;
			if(!sd.typeTarget[0]){
				sd.typeTarget[0]=tp[i].pTarget;
				sd.powerLimit[0]=tp[i].pTarget/2;
			}
		}
		else if(tplist[i] % 12 == 5){
			++sd.tpcnt[1];
			tp[i].type = 1;
			tp[i].pTarget = (tplist[i]-1)/4;
			if(!sd.typeTarget[1]){
				sd.typeTarget[1]=tp[i].pTarget;
				sd.powerLimit[1]=tp[i].pTarget/2;
			}
		}
		else if(tplist[i] % 12 == 11){
			++sd.tpcnt[2];
			tp[i].type = 2;
			tp[i].pTarget = (tplist[i]-1)/2;
			if(!sd.typeTarget[2]){
				sd.typeTarget[2]=tp[i].pTarget;
				sd.powerLimit[2]=tp[i].pTarget/2;
			}
		}
		else{
			fprintf(stderr, "error during setup of test prime array\n");
			printf( "error during setup of test prime array\n");
			exit(EXIT_FAILURE);
		}
	}

	if(boinc_is_standalone()){
		printf("Testing %u primes.  There are %u 1 mod 3 primes, %u 5 mod 12 primes, and %u 11 mod 12 primes\n",st.tpcount,sd.tpcnt[0],sd.tpcnt[1],sd.tpcnt[2]);
		printf("Factorial targets are %" PRIu64 ", %" PRIu64 ", %" PRIu64 "\n",sd.typeTarget[0],sd.typeTarget[1],sd.typeTarget[2]);
		printf("     Power limits are %" PRIu64 ", %" PRIu64 ", %" PRIu64 "\n",sd.powerLimit[0],sd.powerLimit[1],sd.powerLimit[2]);		
	}
	fprintf(stderr, "Testing %u primes.  There are %u 1 mod 3 primes, %u 5 mod 12 primes, and %u 11 mod 12 primes\n",st.tpcount,sd.tpcnt[0],sd.tpcnt[1],sd.tpcnt[2]);

	pd.d_testprime = clCreateBuffer( hardware.context, CL_MEM_READ_WRITE, st.tpcount*sizeof(cl_ulong), NULL, &err );
	if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure.\n");
		printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}
	pd.d_testprimedata = clCreateBuffer( hardware.context, CL_MEM_READ_WRITE, st.tpcount*sizeof(cl_ulong8), NULL, &err );
	if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure.\n");
		printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}
	pd.d_residues = clCreateBuffer( hardware.context, CL_MEM_READ_WRITE, st.tpcount*sizeof(cl_ulong2), NULL, &err );
	if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure.\n");
		printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}

	// send test primes to gpu, blocking
	sclWrite(hardware, st.tpcount * sizeof(cl_ulong), pd.d_testprime, tplist);
	free(tplist);

	// kernel used in profileGPU, setup arg
	sclSetKernelArg(pd.clearresult, 0, sizeof(cl_mem), &pd.d_primecount);
	sclSetKernelArg(pd.clearresult, 1, sizeof(cl_mem), &pd.d_totalcount);
	sclSetGlobalSize( pd.clearresult, 1 );

	profileGPU(pd,sd,hardware);
	
	sclSetGlobalSize( pd.mulsmall, sd.psize/4 );
	sclSetGlobalSize( pd.mullarge, sd.psize/4 );

//	printf("global size for mul %" PRIu64 "\n",pd.mulsmall.global_size[0]);

	sd.numgroups = pd.mulsmall.global_size[0]/256;
	
//	printf("numgroups %u\n",sd.numgroups);	

	sclSetGlobalSize( pd.getsegprps, sd.range/60+1 );

//	printf("getsegprps gs %" PRIu64"\n",pd.getsegprps.global_size[0]);
	
	const uint32_t itersize = 2560000;
	sclSetGlobalSize( pd.iterate, itersize );
	const uint32_t itergroups = (sd.nvidia) ? pd.iterate.global_size[0]/1024 : pd.iterate.global_size[0]/256;	
	
	sclSetGlobalSize( pd.clearn, 1 );
	sclSetGlobalSize( pd.clearacu, 1 );
	
	const uint32_t stride = 256000;
	sclSetGlobalSize( pd.setup, stride );
	sclSetGlobalSize( pd.finda, stride );
	sclSetGlobalSize( pd.findc, stride );
	sclSetGlobalSize( pd.findu, stride );

	pd.d_primes = clCreateBuffer(hardware.context, CL_MEM_READ_WRITE, sd.psize*sizeof(cl_ulong), NULL, &err);
        if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure d_primes\n");
                printf( "ERROR: clCreateBuffer failure d_primes\n" );
		exit(EXIT_FAILURE);
	}
	for(int i=0; i<3; ++i){
		pd.d_powers[i] = clCreateBuffer(hardware.context, CL_MEM_READ_WRITE, sd.psize*sizeof(cl_uint2), NULL, &err);
		if ( err != CL_SUCCESS ) {
			fprintf(stderr, "ERROR: clCreateBuffer failure d_powers\n");
			printf( "ERROR: clCreateBuffer failure d_powers\n" );
			exit(EXIT_FAILURE);
		}
	}

	pd.d_grptotal = clCreateBuffer(hardware.context, CL_MEM_READ_WRITE, sd.numgroups*sizeof(cl_ulong2), NULL, &err);
	if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure d_grptotal\n");
		printf( "ERROR: clCreateBuffer failure d_grptotal\n" );
		exit(EXIT_FAILURE);
	}

	pd.d_found = clCreateBuffer( hardware.context, CL_MEM_READ_WRITE, sizeof(cl_uint), NULL, &err );
	if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure.\n");
		printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}
	pd.d_acu = clCreateBuffer( hardware.context, CL_MEM_READ_WRITE, ACUBUFFER * sizeof(cl_ulong), NULL, &err );
	if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure.\n");
		printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}
	
	// set static kernel args
	sclSetKernelArg(pd.clearn, 0, sizeof(cl_mem), &pd.d_primecount);
	sclSetKernelArg(pd.clearn, 1, sizeof(cl_mem), &pd.d_totalcount);	
		
	sclSetKernelArg(pd.getsegprps, 3, sizeof(cl_mem), &pd.d_primes);
	sclSetKernelArg(pd.getsegprps, 4, sizeof(cl_mem), &pd.d_primecount);
	sclSetKernelArg(pd.getsegprps, 5, sizeof(cl_mem), &pd.d_powers[0]);
	sclSetKernelArg(pd.getsegprps, 6, sizeof(cl_mem), &pd.d_powers[1]);
	sclSetKernelArg(pd.getsegprps, 7, sizeof(cl_mem), &pd.d_powers[2]);
	sclSetKernelArg(pd.getsegprps, 8, sizeof(uint64_t), &sd.typeTarget[0]);
	sclSetKernelArg(pd.getsegprps, 9, sizeof(uint64_t), &sd.typeTarget[1]);
	sclSetKernelArg(pd.getsegprps, 10, sizeof(uint64_t), &sd.typeTarget[2]);
	sclSetKernelArg(pd.getsegprps, 11, sizeof(uint64_t), &sd.powerLimit[0]);
	sclSetKernelArg(pd.getsegprps, 12, sizeof(uint64_t), &sd.powerLimit[1]);
	sclSetKernelArg(pd.getsegprps, 13, sizeof(uint64_t), &sd.powerLimit[2]);

	sclSetKernelArg(pd.clearacu, 0, sizeof(cl_mem), &pd.d_found);

	sclSetKernelArg(pd.finda, 0, sizeof(cl_mem), &pd.d_found);
	sclSetKernelArg(pd.finda, 1, sizeof(cl_mem), &pd.d_acu);

	sclSetKernelArg(pd.findc, 0, sizeof(cl_mem), &pd.d_found);
	sclSetKernelArg(pd.findc, 1, sizeof(cl_mem), &pd.d_acu);

	sclSetKernelArg(pd.findu, 0, sizeof(cl_mem), &pd.d_found);
	sclSetKernelArg(pd.findu, 1, sizeof(cl_mem), &pd.d_acu);

	sclSetKernelArg(pd.reduce, 0, sizeof(cl_mem), &pd.d_testprimedata);
	sclSetKernelArg(pd.reduce, 1, sizeof(cl_mem), &pd.d_residues);
	sclSetKernelArg(pd.reduce, 2, sizeof(cl_mem), &pd.d_grptotal);
	sclSetKernelArg(pd.reduce, 4, sizeof(uint32_t), &sd.numgroups);
	
	sclSetKernelArg(pd.mulsmall, 0, sizeof(cl_mem), &pd.d_testprimedata);
	sclSetKernelArg(pd.mulsmall, 3, sizeof(cl_mem), &pd.d_grptotal);	

	sclSetKernelArg(pd.mullarge, 0, sizeof(cl_mem), &pd.d_testprimedata);
	sclSetKernelArg(pd.mullarge, 1, sizeof(cl_mem), &pd.d_primes);
	sclSetKernelArg(pd.mullarge, 2, sizeof(cl_mem), &pd.d_primecount);
	sclSetKernelArg(pd.mullarge, 4, sizeof(cl_mem), &pd.d_grptotal);

	sd.maxtarget = sd.typeTarget[2];
	if(sd.maxtarget < sd.typeTarget[1]) sd.maxtarget = sd.typeTarget[1];
	if(sd.maxtarget < sd.typeTarget[0]) sd.maxtarget = sd.typeTarget[0];

	uint32_t resume = 0;

	if( sd.test ){
		// clear result file
		FILE * temp_file = my_fopen(RESULT_FILENAME,"w");
		if (temp_file == NULL){
			fprintf(stderr,"Cannot open %s !!!\n",RESULT_FILENAME);
			exit(EXIT_FAILURE);
		}
		fclose(temp_file);
	}
	else{
		int rsr = read_state(sd, st, residues);

		if( rsr == 2 ){
			// trying to resume a finished workunit
			if(boinc_is_standalone()){
				printf("Workunit complete.\n");
			}
			fprintf(stderr,"Workunit complete.\n");
			boinc_finish(EXIT_SUCCESS);
		}
		else if( rsr == 1 ){
			// resuming
			if(boinc_is_standalone()){
				printf("Resuming search from checkpoint. Current P: %" PRIu64 "\n", st.currp);
			}
			fprintf(stderr,"Resuming search from checkpoint. Current P: %" PRIu64 "\n", st.currp);
			// send residues to gpu, blocking
			sclWrite(hardware, st.tpcount * sizeof(cl_ulong2), pd.d_residues, residues);
			resume = 1;
		}
		else{
			// starting from beginning
			// clear result file
			FILE * temp_file = my_fopen(RESULT_FILENAME,"w");
			if (temp_file == NULL){
				fprintf(stderr,"Cannot open %s !!!\n",RESULT_FILENAME);
				exit(EXIT_FAILURE);
			}
			fclose(temp_file);

			// setup boinc trickle up
			st.trickle = (uint64_t)time(NULL);
		}
	}
	
	// for small prime generation on cpu
	bool freed = true;	
	primesieve_iterator it;
	uint64_t * smprime = NULL;
	cl_ulong2 * smpower = NULL;
	cl_ulong * h_prime = NULL;
	cl_ulong2 * h_power = NULL;
	
	if(st.currp < 0xFFFFFFFF){
		freed = false;
		primesieve_init(&it);
		primesieve_jump_to(&it, st.currp, 0xFFFFFFFF);
		
		// beginning 32 bit host prime and power tables
		smprime = (uint64_t *)malloc(sd.psize*sizeof(uint64_t));
		if( smprime == NULL ){
			fprintf(stderr,"malloc error: smprime\n");
			exit(EXIT_FAILURE);
		}
		smpower = (cl_ulong2 *)malloc(sd.psize*sizeof(cl_ulong2));
		if( smpower == NULL ){
			fprintf(stderr,"malloc error: smpower\n");
			exit(EXIT_FAILURE);
		}
		// compressed 32 bit host prime and power tables
		h_prime = (cl_ulong *)malloc(sd.psize*sizeof(cl_ulong));
		if( h_prime == NULL ){
			fprintf(stderr,"malloc error: h_prime\n");
			exit(EXIT_FAILURE);
		}
		h_power = (cl_ulong2 *)malloc(sd.psize*sizeof(cl_ulong2));
		if( h_power == NULL ){
			fprintf(stderr,"malloc error: h_power\n");
			exit(EXIT_FAILURE);
		}
		// device 32 bit prime and power tables
		for(int i=0; i<3; ++i){
			pd.d_primes32[i] = clCreateBuffer(hardware.context, CL_MEM_READ_ONLY, sd.psize*sizeof(cl_ulong), NULL, &err);
			if ( err != CL_SUCCESS ) {
				fprintf(stderr, "ERROR: clCreateBuffer failure d_primes32\n");
				printf( "ERROR: clCreateBuffer failure d_primes\n" );
				exit(EXIT_FAILURE);
			}		
			pd.d_powers32[i] = clCreateBuffer(hardware.context, CL_MEM_READ_ONLY, sd.psize*sizeof(cl_ulong2), NULL, &err);
			if ( err != CL_SUCCESS ) {
				fprintf(stderr, "ERROR: clCreateBuffer failure d_powers32\n");
				printf( "ERROR: clCreateBuffer failure d_powers\n" );
				exit(EXIT_FAILURE);
			}
		}		
	}	

	sclEnqueueKernel(hardware, pd.clearresult);

	// setup test prime constants
	sclSetKernelArg(pd.setup, 0, sizeof(cl_mem), &pd.d_testprime);
	sclSetKernelArg(pd.setup, 1, sizeof(cl_mem), &pd.d_testprimedata);
	sclSetKernelArg(pd.setup, 2, sizeof(uint32_t), &st.tpcount);
	sclSetKernelArg(pd.setup, 3, sizeof(uint64_t), &sd.typeTarget[0]);
	sclSetKernelArg(pd.setup, 4, sizeof(uint64_t), &sd.typeTarget[1]);
	sclSetKernelArg(pd.setup, 5, sizeof(uint64_t), &sd.typeTarget[2]);
	sclSetKernelArg(pd.setup, 6, sizeof(cl_mem), &pd.d_residues);
	sclSetKernelArg(pd.setup, 7, sizeof(uint32_t), &resume);
	sclEnqueueKernel(hardware, pd.setup);
	sclReleaseMemObject(pd.d_testprime);

	time(&boinc_last);
	time(&ckpt_last);
	time_t totals, totalf;
	if(boinc_is_standalone()){
		time(&totals);
	}
	uint32_t kernelq = 0;
	cl_event launchEvent = NULL;
	uint32_t maxq = 100;

	// main search loop
	while(st.currp <= sd.maxtarget){

		// free memory after primes < 2^32 are completed
		if(!freed && st.currp > 0xFFFFFFFF){
			freed = true;
			primesieve_free_iterator(&it);
			free(smprime);
			free(smpower);
			free(h_prime);
			free(h_power);
			for(int i=0; i<3; ++i){
				sclReleaseMemObject(pd.d_powers32[i]);
				sclReleaseMemObject(pd.d_primes32[i]);
			}			
		}

		time(&time_curr);
		int ckpt_time = (int)time_curr - (int)ckpt_last;
		if( ckpt_time > 60 ){
			ckpt_last = time_curr;
			getFractionDone(sd, st, 0);				
			// 1 minute checkpoint
			if(kernelq > 0){
				waitOnEvent(hardware, launchEvent);
				kernelq = 0;
			}
			boinc_begin_critical_section();
			getDataFromGPU(pd, sd, hardware, st, residues, h_primecount);
			checkpoint(sd, st, residues, ckpt_time);
			boinc_end_critical_section();
			// clear counters
			sclEnqueueKernel(hardware, pd.clearresult);
		}

		uint64_t stop = getPrimes(hardware, pd, sd, st, smprime, smpower, h_prime, h_power, it);
		double chunksize = (double)(stop - st.currp);

		// group prime types for cache and multiply
		uint32_t tpcnt = 0;
		for(uint32_t j=0; j<3; ++j){
			for(uint32_t i=0; i<st.tpcount; ++i){
				if(tp[i].type != j)
					continue;
				++tpcnt;
				if(st.currp > sd.typeTarget[j])
					continue;
				multiply(hardware, pd, sd, st, i, j);
				sclSetKernelArg(pd.reduce, 3, sizeof(uint32_t), &i);
				if(kernelq == 0){
					launchEvent = sclEnqueueKernelEvent(hardware, pd.reduce);
				}
				else{
					sclEnqueueKernel(hardware, pd.reduce);
				}
				if(++kernelq == maxq){
					time(&time_curr);
					if( ((int)time_curr - (int)boinc_last) > 3 ){
						boinc_last = time_curr;
						// update BOINC fraction done every 4 sec
						double partialDone = (double)tpcnt / (double)st.tpcount * chunksize;
						getFractionDone(sd, st, partialDone);
					}				
					// limit cl queue depth and sleep cpu
					waitOnEvent(hardware, launchEvent);
					kernelq = 0;
				}				
			}
		}
		
		// add kernel prp count to total count and clear kernel prp count
		sclEnqueueKernel(hardware, pd.clearn);	
			
		st.currp = stop;
	}


	if(kernelq > 0){
		waitOnEvent(hardware, launchEvent);
		kernelq=0;
	}

	// iterate from type target factorial to each prime's target factorial
	sclSetKernelArg(pd.iterate, 0, sizeof(cl_mem), &pd.d_testprimedata);
	sclSetKernelArg(pd.iterate, 1, sizeof(cl_mem), &pd.d_residues);	
	for(uint32_t startTp = 0; startTp < st.tpcount; startTp += itergroups){
		sclSetKernelArg(pd.iterate, 2, sizeof(uint32_t), &startTp);
		sclSetKernelArg(pd.iterate, 3, sizeof(uint32_t), &st.tpcount);		
		sclEnqueueKernel(hardware, pd.iterate);
//		float kernel_ms = ProfilesclEnqueueKernel(hardware, pd.iterate);
//		printf("iterate %0.2fms\n",kernel_ms);
	}

	// finalize results
	boinc_begin_critical_section();
	getDataFromGPU(pd, sd, hardware, st, residues, h_primecount);
	getResults(pd, sd, hardware, st, residues, tp);
	finalizeResults(sd);
	st.done = 1;
	boinc_fraction_done(1.0);
	checkpoint(sd, st, residues, 0);
	boinc_end_critical_section();


	fprintf(stderr,"Search complete. Results: %u, total power table primes generated %" PRIu64 ", removed %u PRPs\n", sd.resultcount, st.totalcount-sd.prpsremoved, sd.prpsremoved);

	if(boinc_is_standalone()){
		time(&totalf);
		printf("Search finished in %d sec.\n", (int)totalf - (int)totals);
		printf("results %u, total power table primes generated %" PRIu64 ", checksum %016" PRIX64 ", removed %u PRPs\n", sd.resultcount, st.totalcount-sd.prpsremoved, sd.checksum, sd.prpsremoved);
	}

	free(tp);
	free(residues);
	free(h_primecount);
	cleanup(pd);

}


void resetData(searchData & sd, workStatus & st){
	sd.tpcnt[0] = 0;
	sd.tpcnt[1] = 0;
	sd.tpcnt[2] = 0;
	sd.typeTarget[0] = 0;
	sd.typeTarget[1] = 0;
	sd.typeTarget[2] = 0;
	sd.powerLimit[0] = 0;
	sd.powerLimit[1] = 0;
	sd.powerLimit[2] = 0;
	sd.checksum = 0;
	sd.resultcount = 0;
	sd.prpsremoved = 0;
	st.totalcount = 0;
	sd.testResultPrime = 0;
	sd.testResultValue = 0;
}


void run_test( sclHard hardware, searchData & sd, workStatus & st ){

	int goodtest = 0;

	printf("Beginning self test of 7 ranges.\n\n");

	time_t start, finish;
	time(&start);

//	-p 1239053554603 -P 1239053554604 (type 0)
	st.pmin = 1239053554603ULL;
	st.pmax = 1239053554604ULL;
	printf("1239053554603 is a type 0 prime\n");
	cl_sieve(hardware, sd, st);
	if( sd.resultcount == 1 && sd.checksum == 0x00000240FAB1A752 && st.totalcount-sd.prpsremoved == 8257082014ULL
		&& sd.testResultPrime == 1239053554603ULL && sd.testResultValue == -4 ){
		printf("test case 1 passed.\n\n");
		fprintf(stderr,"test case 1 passed.\n");
		++goodtest;
	}
	else{
		printf("test case 1 failed.\n\n");
		fprintf(stderr,"test case 1 failed.\n");
	}
	resetData(sd,st);

//	-p 1108967825921 -P 1108967825922
	st.pmin = 1108967825921ULL;
	st.pmax = 1108967825922ULL;
	printf("1108967825921 is a type 1 prime\n");
	cl_sieve(hardware, sd, st);
	if( sd.resultcount == 1 && sd.checksum == 0x0000010233A2220D && st.totalcount-sd.prpsremoved == 10956003002ULL
		&& sd.testResultPrime == 1108967825921ULL && sd.testResultValue == 12 ){
		printf("test case 2 passed.\n\n");
		fprintf(stderr,"test case 2 passed.\n");
		++goodtest;
	}
	else{
		printf("test case 2 failed.\n\n");
		fprintf(stderr,"test case 2 failed.\n");
	}
	resetData(sd,st);

//	-p 5609877309359 -P 5609877309360 (type 2)
	st.pmin = 5609877309359ULL;
	st.pmax = 5609877309360ULL;
	printf("5609877309359 is a type 2 prime\n");
	cl_sieve(hardware, sd, st);
	if( sd.resultcount == 1 && sd.checksum == 0x00000A344D7D0F58 && st.totalcount-sd.prpsremoved == 101542897873ULL
		&& sd.testResultPrime == 5609877309359ULL && sd.testResultValue == -6 ){	
		printf("test case 3 passed.\n\n");
		fprintf(stderr,"test case 3 passed.\n");
		++goodtest;
	}
	else{
		printf("test case 3 failed.\n\n");
		fprintf(stderr,"test case 3 failed.\n");
	}
	resetData(sd,st);

//	-p 16556218163369 -P 16556218163370
	st.pmin = 16556218163369ULL;
	st.pmax = 16556218163370ULL;
	printf("16556218163369 is a type 1 prime\n");
	cl_sieve(hardware, sd, st);
	if( sd.resultcount == 1 && sd.checksum == 0x00000F0ECB80A0AB && st.totalcount-sd.prpsremoved == 147755473426ULL
		&& sd.testResultPrime == 16556218163369ULL && sd.testResultValue == 2 ){	
		printf("test case 4 passed.\n\n");
		fprintf(stderr,"test case 4 passed.\n");
		++goodtest;
	}
	else{
		printf("test case 4 failed.\n\n");
		fprintf(stderr,"test case 4 failed.\n");
	}
	resetData(sd,st);
	
//	-p 200 -P 564
	st.pmin = 200;
	st.pmax = 564;
	printf("Testing small iterations with Wilson prime 563\n");	
	cl_sieve(hardware, sd, st);
	if( sd.resultcount == 57 && sd.checksum == 0x00000000000080A3 && st.totalcount-sd.prpsremoved == 30
		&& sd.testResultPrime == 563 && sd.testResultValue == 0 ){	
		printf("test case 5 passed.\n\n");
		fprintf(stderr,"test case 5 passed.\n");
		++goodtest;
	}
	else{
		printf("test case 5 failed.\n\n");
		fprintf(stderr,"test case 5 failed.\n");
	}
	resetData(sd,st);
	
//	-p 86000000 -P 87467200
	st.pmin = 86000000;
	st.pmax = 87467200;
	printf("Testing large iterations with type 2 prime 87467099\n");	
	cl_sieve(hardware, sd, st);
	if( sd.resultcount == 1 && sd.checksum == 0x0000097C61AB0943 && st.totalcount-sd.prpsremoved == 2604536
		&& sd.testResultPrime == 87467099 && sd.testResultValue == -2 ){	
		printf("test case 6 passed.\n\n");
		fprintf(stderr,"test case 6 passed.\n");
		++goodtest;
	}
	else{
		printf("test case 6 failed.\n\n");
		fprintf(stderr,"test case 6 failed.\n");
	}
	resetData(sd,st);
	
//	-p 17524177394450 -P 17524177394618
	st.pmin = 17524177394450ULL;
	st.pmax = 17524177394618ULL;
	printf("17524177394617 is a type 0 prime\n");	
	cl_sieve(hardware, sd, st);
	if( sd.resultcount == 1 && sd.checksum == 0x00005B54B4CBBC47 && st.totalcount-sd.prpsremoved == 304620766446
		&& sd.testResultPrime == 17524177394617 && sd.testResultValue == 256 ){	
		printf("test case 7 passed.\n\n");
		fprintf(stderr,"test case 7 passed.\n");
		++goodtest;
	}
	else{
		printf("test case 7 failed.\n\n");
		fprintf(stderr,"test case 7 failed.\n");
	}
	resetData(sd,st);

//	done
	if(goodtest == 7){
		printf("All test cases completed successfully!\n");
		fprintf(stderr, "All test cases completed successfully!\n");
	}
	else{
		printf("Self test FAILED!\n");
		fprintf(stderr, "Self test FAILED!\n");
	}

	time(&finish);
	printf("Elapsed time: %d sec.\n", (int)finish - (int)start);

}







