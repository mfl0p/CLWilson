/*

	clearn.cl - Bryan Little Jul 2025

	adds kernel prp count to total
	keeps track of largest kernel prp count for array bounds check on cpu
	clears prp counter

*/


__kernel void clearn(__global uint *g_primecount, __global ulong *g_totalcount){

	const uint gid = get_global_id(0);

	if(gid == 0){
		const uint pcnt = g_primecount[0];
		
		g_totalcount[0] += pcnt;
		
		if( pcnt > g_primecount[1] ){
			g_primecount[1] = pcnt;
		}
		
		g_primecount[0] = 0;
	}


}



