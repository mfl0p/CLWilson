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
#include <getopt.h>
#include <cinttypes>

#include "boinc_api.h"
#include "boinc_opencl.h"
#include "simpleCL.h"
#include "primesieve.h"
#include "putil.h"
#include "cl_wilson.h"

void help()
{
	printf("Welcome to CLWilson, an OpenCL program to search for Wilson Primes\n");
	printf("Program usage:\n");  
	printf("-p #	Starting prime to search p\n");
	printf("-P #	End prime prime to search P, range [-p, -P) exclusive, 5 <= -p <= p < -P <= %" PRIu64 "\n", maxp);
	printf("	Required range is <= 10e6\n");
	printf("-s 	Perform self test to verify proper operation of the program with the current GPU.\n");
	printf("-h	Print this help\n");
        boinc_finish(EXIT_FAILURE);
}


static const char *short_opts = "p:P:sd:h";

static int parse_option(int opt, char *arg, const char *source, workStatus *st, searchData *sd)
{
  int status = 0;

  switch (opt)
  {
    case 'p':
      status = parse_uint64(&st->pmin,arg,5,maxp-1);
      break;

    case 'P':
      status = parse_uint64(&st->pmax,arg,6,maxp);
      break;
      
    case 's':
      sd->test = true;
      fprintf(stderr,"Performing self test.\n");
      printf("Performing self test.\n");
      break;

    case 'd':
      break;

    case 'h':
      help();
      break;

    case '?':
      help();
      break;
  }

  return status;
}

static const struct option long_opts[] = {
  {"device",  optional_argument, 0, 'd'},		// handle --device arg, but it's not used
  {"test",  no_argument, 0, 's'},
  {0,0,0,0}
};


/* Process command-line options using getopt_long().
   Non-option arguments are treated as if they belong to option zero.
   Returns the number of options processed.
 */
static int process_args(int argc, char *argv[], workStatus *st, searchData *sd)
{
  int count = 0, ind = -1, opt;

  while ((opt = getopt_long(argc,argv,short_opts,long_opts,&ind)) != -1)
    switch (parse_option(opt,optarg,NULL,st,sd))
    {
      case 0:
        ind = -1;
        count++;
        break;

      case -1:
        /* If ind is unchanged then this is a short option, otherwise long. */
        if (ind == -1){
          printf("%s: invalid argument -%c %s\n",argv[0],opt,optarg);
          fprintf(stderr,"%s: invalid argument -%c %s\n",argv[0],opt,optarg);
	}
        else{
     	  printf("%s: invalid argument --%s %s\n",argv[0],long_opts[ind].name,optarg);
          fprintf(stderr,"%s: invalid argument --%s %s\n",argv[0],long_opts[ind].name,optarg);
	}
        boinc_finish(EXIT_FAILURE);

      case -2:
        /* If ind is unchanged then this is a short option, otherwise long. */
        if (ind == -1){
          printf("%s: out of range argument -%c %s\n",argv[0],opt,optarg);
          fprintf(stderr,"%s: out of range argument -%c %s\n",argv[0],opt,optarg);
	}
        else{
          printf("%s: out of range argument --%s %s\n",argv[0],long_opts[ind].name,optarg);
          fprintf(stderr,"%s: out of range argument --%s %s\n",argv[0],long_opts[ind].name,optarg);
	}
        boinc_finish(EXIT_FAILURE);

      default:
        printf("unknown command line argument\n");
        boinc_finish(EXIT_FAILURE);
    }

  while (optind < argc)
    switch (parse_option(0,argv[optind],NULL,st,sd))
    {
      case 0:
        optind++;
        count++;
        break;

      case -1:
        fprintf(stderr,"%s: invalid non-option argument %s\n",argv[0],argv[optind]);
        boinc_finish(EXIT_FAILURE);

      case -2:
        fprintf(stderr,"%s: out of range non-option argument %s\n",argv[0],argv[optind]);
        boinc_finish(EXIT_FAILURE);

      default:
        boinc_finish(EXIT_FAILURE);
    }


  return count;
}


int main(int argc, char *argv[])
{ 
	sclHard hardware;
	searchData sd = {};
	workStatus st = {};
	sd.write_state_a_next = true;

        // Initialize BOINC
        BOINC_OPTIONS options;
        boinc_options_defaults(options);
        options.normal_thread_priority = true;
        boinc_init_options(&options);

	fprintf(stderr, "\nCLWilson v%s.%s by Bryan Little\nwith contributions by Yves Gallot, and Kim Walisch\n",VERSION_MAJOR,VERSION_MINOR);
	fprintf(stderr, "Compiled " __DATE__ " with GCC " __VERSION__ "\n");
	if(boinc_is_standalone()){
		printf("\nCLWilson v%s.%s by Bryan Little\nwith contributions by Yves Gallot, and Kim Walisch\n",VERSION_MAJOR,VERSION_MINOR);
		printf("Compiled " __DATE__ " with GCC " __VERSION__ "\n");
	}

        // Print out cmd line for diagnostics
        fprintf(stderr, "Command line: ");
        for (int i = 0; i < argc; i++)
        	fprintf(stderr, "%s ", argv[i]);
        fprintf(stderr, "\n");

	process_args(argc,argv,&st,&sd);

	primesieve_set_num_threads(1);

	cl_platform_id platform = 0;
	cl_device_id device = 0;
	cl_context ctx;
	cl_command_queue queue;
	cl_int err = 0;

	int retval = 0;
	retval = boinc_get_opencl_ids(argc, argv, 0, &device, &platform);
	if (retval) {
		if(boinc_is_standalone()){
			printf("init_data.xml not found, using device 0.\n");

			err = clGetPlatformIDs(1, &platform, NULL);
			if (err != CL_SUCCESS) {
				printf( "clGetPlatformIDs() failed with %d\n", err );
				fprintf(stderr, "Error: clGetPlatformIDs() failed with %d\n", err );
				exit(EXIT_FAILURE);
			}
			err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
			if (err != CL_SUCCESS) {
				printf( "clGetDeviceIDs() failed with %d\n", err );
				fprintf(stderr, "Error: clGetDeviceIDs() failed with %d\n", err );
				exit(EXIT_FAILURE);
			}
		}
		else{
			fprintf(stderr, "Error: boinc_get_opencl_ids() failed with error %d\n", retval );
			exit(EXIT_FAILURE);
		}
	}

	cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };

	ctx = clCreateContext(cps, 1, &device, NULL, NULL, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Error: clCreateContext() returned %d\n", err);
        	exit(EXIT_FAILURE); 
   	}

	// OpenCL v2.0
	//cl_queue_properties qp[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
	//queue = clCreateCommandQueueWithProperties(ctx, device, qp, &err);

	queue = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);	
	if(err != CL_SUCCESS) { 
		fprintf(stderr, "Error: Creating Command Queue. (clCreateCommandQueueWithProperties) returned %d\n", err );
		exit(EXIT_FAILURE);
    	}

	hardware.platform = platform;
	hardware.device = device;
	hardware.queue = queue;
	hardware.context = ctx;

 	char device_name[1024];
 	char device_vend[1024];
 	char device_driver[1024];
	cl_uint CUs;
	cl_ulong max_malloc, global_mem;

	err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), &device_name, NULL);
	if (err != CL_SUCCESS) {
		printf( "clGetDeviceInfo failed with %d\n", err );
		exit(EXIT_FAILURE);
	}
	err = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(device_vend), &device_vend, NULL);
	if (err != CL_SUCCESS) {
		printf( "clGetDeviceInfo failed with %d\n", err );
		exit(EXIT_FAILURE);
	}
	err = clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(device_driver), &device_driver, NULL);
	if (err != CL_SUCCESS) {
		printf( "clGetDeviceInfo failed with %d\n", err );
		exit(EXIT_FAILURE);
	}
	err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &CUs, NULL);
	if (err != CL_SUCCESS) {
		printf( "clGetDeviceInfo failed with %d\n", err );
		exit(EXIT_FAILURE);
	}
	err = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_malloc, NULL);
	if (err != CL_SUCCESS) {
		printf( "clGetDeviceInfo failed with %d\n", err );
		exit(EXIT_FAILURE);
	}
	err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem, NULL);
	if (err != CL_SUCCESS) {
		printf( "clGetDeviceInfo failed with %d\n", err );
		exit(EXIT_FAILURE);
	}
	sd.maxmalloc = (int64_t)max_malloc;
	sd.globalmem = (int64_t)global_mem;

	fprintf(stderr, "GPU Info:\n  Name: \t\t%s\n  Vendor: \t\t%s\n  Driver: \t\t%s\n  Compute Units: \t%u\n", device_name, device_vend, device_driver, CUs);
	if(boinc_is_standalone()){
		printf("GPU Info:\n  Name: \t\t%s\n  Vendor: \t\t%s\n  Driver: \t\t%s\n  Compute Units: \t%u\n", device_name, device_vend, device_driver, CUs);
	}

	// check vendor and normalize compute units
	// kernel size will be determined by profiling so this doesn't have to be accurate.
	sd.computeunits = CUs;
	char intel_s[] = "Intel";
	char arc_s[] = "Arc";
	char nvidia_s[] = "NVIDIA";	

	if(strstr((char*)device_vend, (char*)nvidia_s) != NULL){
		sd.nvidia = true;
	}
	// Intel
	else if( strstr((char*)device_vend, (char*)intel_s) != NULL ){

		if( strstr((char*)device_name, (char*)arc_s) != NULL ){
			sd.computeunits /= 10;
		}
		else{
			sd.computeunits /= 20;
	                fprintf(stderr,"Detected Intel integrated graphics\n");	
		}

	}
	// AMD
        else{
		sd.computeunits /= 2;
        }

	if(!sd.computeunits) sd.computeunits = 1;

	if(sd.test){
		run_test(hardware, sd, st);
	}
	else{
		cl_sieve(hardware, sd, st);
	}

        sclReleaseClHard(hardware);

	boinc_finish(EXIT_SUCCESS);

	return 0; 
} 

