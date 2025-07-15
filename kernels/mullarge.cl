/* 
	mullarge.cl -- Bryan Little, Yves Gallot, Jul 2025

	Wilson search OpenCL Kernel 
	
	multiply by prime^power for each prime >2^32
	
	these primes/powers (actually prps) are generated on GPU
	
	limit is used to reduce memory access and power calculation when we know power = 1

*/


__kernel __attribute__ ((reqd_work_group_size(256, 1, 1))) void mullarge(
				__global ulong8 *g_tpdata,
				__global ulong *g_prime,
				__global uint *g_primecount,
				__global uint2 *g_power,
				__global ulong2 *g_grptotal,
				const uint tpnum,
				const ulong limit,
				const ulong target )
{
	const uint gid = get_global_id(0);
	const uint lid = get_local_id(0);
	const uint gs = get_global_size(0);
	const uint pcnt = g_primecount[0];
	__local ulong2 total[256];

	// s0=p s1=q s2=one.s0 s3=one.s1 s4=r2.s0 s5=r2.s1 s6=target factorial for this type s7=target factorial for this prime
	const ulong8 tp = g_tpdata[tpnum];
	total[lid] = (ulong2)(tp.s2, tp.s3);		// set to one
	bool first_iter = true;

	for(uint i = gid; i < pcnt; i+= gs){
		ulong prime = g_prime[i];
		uint2 power = (prime > limit) ? (uint2)(1,0) : g_power[i];
		if(prime <= target){
			const ulong2 base = m2p_mul_r2( prime, (ulong2)(tp.s4, tp.s5), tp.s0, tp.s1);	// convert prime to montgomery form
			ulong2 primepow;
			if(power.s0 == 1){
				primepow = base;
			}
			else{
				ulong2 a = base;
				while( power.s1 ){
					a = m2p_square(a, tp.s0, tp.s1);
					if(power.s0 & power.s1){
						a = m2p_mul(a, base, tp.s0, tp.s1);
					}
					power.s1 >>= 1;
				}
				primepow = a;
			}
			if(first_iter){
				first_iter = false;
				total[lid] = primepow;
			}
			else{
				total[lid] = m2p_mul(total[lid], primepow, tp.s0, tp.s1);
			}
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for(uint s = 128; s > 0; s >>= 1){
		if(lid < s){
			total[lid] = m2p_mul(total[lid], total[lid+s], tp.s0, tp.s1);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(lid == 0){
		g_grptotal[get_group_id(0)] = total[0];
	}

}





