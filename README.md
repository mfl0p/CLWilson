# CLWilson

CLWilson by Bryan Little

A BOINC enabled OpenCL program to search for Wilson Primes (p − 1)! = −1 (mod p^2)

With contributions by
* Yves Gallot
* Kim Walisch

## Requirements

* OpenCL v1.1
* 64 bit operating system

## How it works

1. Search parameters are given on the command line.
2. Test primes are split into 3 types, 1 mod 3 primes, 5 mod 12 primes, and 11 mod 12 primes.
3. The 3 types allow fast calculation of (p-1)! mod p^2.
4. Report results in results.txt, along with a checksum at the end.
5. Checksum can be used to compare results in a BOINC quorum.

## Running the program
```
Note: prps.dat file is required to be in the same directory as the program.

command line options
* -p #	Starting prime to search p
* -P #	End prime prime to search P, range [-p, -P) exclusive, 5 <= -p <= p < -P <= 4611686018427387903
	Required range is <= 10e6
* -s 	Perform self test to verify proper operation of the program with the current GPU.
* -r 	Verify all results (up to 2e13) where |w_p/p| < 1/50000 with known good file goodWilsonResults.txt
	-s and -r are for use in standalone testing.
* -h	Print help.

For known good result file info see:
EDGAR COSTA, ROBERT GERBICZ, AND DAVID HARVEY. A SEARCH FOR WILSON PRIMES. 2014.

Program gets the OpenCL GPU device index from BOINC.  To run stand-alone, the program will
default to GPU 0 unless an init_data.xml is in the same directory with the format:

<app_init_data>
<gpu_type>NVIDIA</gpu_type>
<gpu_device_num>0</gpu_device_num>
</app_init_data>

or

<app_init_data>
<gpu_type>ATI</gpu_type>
<gpu_device_num>0</gpu_device_num>
</app_init_data>
```

## Related Links
* [Yves Gallot on GitHub](https://github.com/galloty)
* [primesieve by Kim Walisch](https://github.com/kimwalisch/primesieve)
