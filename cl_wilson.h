
// cl_wilson.h

#define RESULT_FILENAME "results.txt"
#define STATE_FILENAME_A "stateA.ckp"
#define STATE_FILENAME_B "stateB.ckp"
#define GOOD_RES_FILENAME "goodWilsonResults.txt"

const uint64_t maxp = 0xFFFFFFFFFFFFFFFF / 4;

typedef struct {
	uint64_t p;
	uint64_t pTarget;
	uint32_t type;
}testPrime;

typedef struct {
	uint64_t p;
	int32_t v;
}goodResult;

typedef struct {
	uint64_t pmin, pmax, currp, trickle, state_sum, totalcount;
	uint32_t tpcount, done;
}workStatus;

typedef struct {
	double lastp;
	uint64_t checksum;
	uint64_t typeTarget[3];
	uint64_t powerLimit[3];
	uint64_t maxtarget;
	uint64_t testResultPrime;
	int64_t maxmalloc;
	int64_t globalmem;
	uint32_t pcount32[3];
	uint32_t nstep;
	uint32_t sstep;
	uint32_t tpcnt[3];
	uint32_t range;
	uint32_t psize;
	uint32_t numgroups;
	uint32_t resultcount;
	uint32_t prpsremoved;
	uint32_t grescount;
	int32_t computeunits;
	int32_t testResultValue;
	bool write_state_a_next;
	bool test;
	bool resultTest;
	bool nvidia;
}searchData;

typedef struct {
	cl_mem d_prps;
	cl_mem d_primecount;
	cl_mem d_totalcount;
	cl_mem d_primes;
	cl_mem d_powers[3];
	cl_mem d_primes32[3];
	cl_mem d_powers32[3];
	cl_mem d_grptotal;
	cl_mem d_testprime;
	cl_mem d_testprimedata;
	cl_mem d_residues;
	cl_mem d_found;
	cl_mem d_acu;
	sclSoft iterate, clearn, clearresult, setup, getsegprps, mulsmall, mullarge, reduce, finda, findc, findu, clearacu;
}progData;

void cl_wilson( sclHard hardware, searchData & sd, workStatus & st );

void run_test( sclHard hardware, searchData & sd, workStatus & st );
