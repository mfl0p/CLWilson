/* 
	reduce.cl -- Bryan Little, Yves Gallot, Jul 2025

	Wilson search OpenCL Kernel 

	reduce this prime's group totals results from mul kernel to a single ulong2

*/


__kernel __attribute__ ((reqd_work_group_size(LSIZE, 1, 1))) void reduce(
				__global ulong8 *g_tpdata,
				__global ulong2 *g_residues,
				__global ulong2 *g_grptotal,
				const uint tpnum,
				const uint groups_per_p ){
				
	const uint lid = get_local_id(0);
	__local ulong2 total[LSIZE];

	// s0=p s1=q s2=one.s0 s3=one.s1 s4=r2.s0 s5=r2.s1 s6=target factorial for this type s7=target factorial for this prime
	const ulong8 tp = g_tpdata[tpnum];
	ulong2 thread_total = (lid < groups_per_p) ?  g_grptotal[lid] : (ulong2)(tp.s2, tp.s3);

	for(uint j=lid+LSIZE; j<groups_per_p; j+=LSIZE){
		thread_total = m2p_mul( thread_total, g_grptotal[j], tp.s0, tp.s1 );
	}

	total[lid] = thread_total;

	barrier(CLK_LOCAL_MEM_FENCE);

	for(uint s = LSIZE>>1; s > 0; s >>= 1){
		if(lid < s){
			total[lid] = m2p_mul(total[lid], total[lid+s], tp.s0, tp.s1);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(lid == 0){
		g_residues[tpnum] = m2p_mul( g_residues[tpnum], total[0], tp.s0, tp.s1 );
	}
	
}



