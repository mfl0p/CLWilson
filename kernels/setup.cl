/* 
	setup.cl -- Bryan Little, Yves Gallot, Jul 2025

	Wilson search OpenCL Kernel 

	setup test prime constants

*/


__kernel void setup(__global ulong *g_testprime, __global ulong8 *g_tpdata,
			const uint tpcount, const ulong tar0, const ulong tar1, const ulong tar2,
			__global ulong2 *g_residues, const uint resume){

	const uint gid = get_global_id(0);
	const uint gs = get_global_size(0);

	for(uint position = gid; position < tpcount; position+=gs){

		ulong p = g_testprime[position];
		ulong q = invert(p);
		ulong2 one = m2p_one(p);
		ulong2 two = m2p_dup(one, p);
		ulong2 r2 = m2p_dup(two, p);
		r2 = m2p_square(r2, p, q);
		r2 = m2p_square(r2, p, q);
		r2 = m2p_square(r2, p, q);
		r2 = m2p_square(r2, p, q);
		r2 = m2p_square(r2, p, q);	// 4^{2^5} = 2^64

		ulong targettype, targetprime;
		if(p % 3 == 1){
			targettype = tar0;
			targetprime = (p-1)/6;
		}
		else if(p % 12 == 5){
			targettype = tar1;
			targetprime = (p-1)/4;
		}
		else if(p % 12 == 11){
			targettype = tar2;
			targetprime = (p-1)/2;
		}
		// s0=p s1=q s2=one.s0 s3=one.s1 s4=r2.s0 s5=r2.s1 s6=target factorial for this type s7=target factorial for this prime
		g_tpdata[position] = (ulong8)( p, q, one.s0, one.s1, r2.s0, r2.s1, targettype, targetprime );

		if(!resume){
			g_residues[position] = (ulong2)( one.s0, one.s1 );
		}

	}
}














