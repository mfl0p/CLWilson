/* 
	iterate.cl -- Bryan Little, Yves Gallot, Jul 2025

	Wilson search OpenCL Kernel 

	iterate from type target to each test prime's target factorial

*/


__kernel __attribute__ ((reqd_work_group_size(LSIZE, 1, 1))) void iterate(
			__global ulong8 *g_tpdata,
			__global ulong2 *g_tpdataext,
			const uint start,
			const uint stop ){

	const uint lid = get_local_id(0);				
	const uint group = get_group_id(0);
	__local ulong2 total[LSIZE];	
	const uint i = start + group;
	
	// one testprime for each workgroup
	if(i < stop){

		// s0=p s1=q s2=one.s0 s3=one.s1 s4=r2.s0 s5=r2.s1 s6=target factorial for this type s7=target factorial for this prime
		const ulong8 tp = g_tpdata[i];

		total[lid] = (ulong2)(tp.s2, tp.s3);							// set to 1
		bool first_iteration = true;
		ulong currN = tp.s6+1+lid;
		ulong2 McurrN = m2p_mul_r2( currN, (ulong2)(tp.s4, tp.s5), tp.s0, tp.s1);		// convert currN to montgomery form
		const ulong2 MLSIZE = m2p_mul_r2( LSIZE, (ulong2)(tp.s4, tp.s5), tp.s0, tp.s1);	// convert LSIZE to montgomery form
		
		for(; currN <= tp.s7; currN += LSIZE){						// iterate from type target to prime target
			if(first_iteration){
				first_iteration = false;
				total[lid] = McurrN;
			}
			else{
				total[lid] = m2p_mul( McurrN, total[lid], tp.s0, tp.s1);			
			}
			McurrN = m2p_add( McurrN, MLSIZE, tp.s0 );					// add LSIZE
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		for(uint s = LSIZE>>1; s > 0; s >>= 1){
			if(lid < s){
				total[lid] = m2p_mul(total[lid], total[lid+s], tp.s0, tp.s1);
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}

		if(lid == 0){
			total[0] = m2p_mul(total[0], g_tpdataext[i], tp.s0, tp.s1)	;		// continue from last residue
			total[0] = m2p_get(total[0], tp.s0, tp.s1);					// final residue converted from montgomery form
			g_tpdataext[i] = total[0];							// store to global
		}


	}


}







