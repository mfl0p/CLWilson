/*

	clearresult.cl - Bryan Little Jul 2025
	
	clear prp counters
*/


__kernel void clearresult(__global uint *g_primecount, __global ulong *g_totalcount){

	const uint gid = get_global_id(0);

	if(gid == 0){
		g_primecount[0] = 0;  	// kernel prp counter
		g_primecount[1] = 0;	// largest kernel prp count to check array overflow
		g_primecount[2] = 0;	// flag set for getsegprps local memory overflow

		g_totalcount[0] = 0;	// total number of prps generated on gpu
	}

}

