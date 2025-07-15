/* 
	find.cl -- Bryan Little, Dec 2024

	Wilson search OpenCL Kernel portion

	find a, c, or u for use in CPU result calculation

*/

#define ACUBUFFER 100


// find integer square root
ulong isqrt(ulong n) {

	// X_(n+1)
	ulong x = n;

	// c_n
	ulong c = 0;

	// d_n which starts at the highest power of four <= n
	ulong d = 1ULL << 62;			// The second-to-top bit is set.
						// Same as ((unsigned) INT32_MAX + 1) / 2.
	while (d > n) {
		d >>= 2;
	}

	// for dₙ … d₀
	while (d != 0) {
		if(x >= c + d){		// if X_(m+1) ≥ Y_m then a_m = 2^m
			x -= c + d;		// X_m = X_(m+1) - Y_m
			c = (c >> 1) + d;	// c_(m-1) = c_m/2 + d_m (a_m is 2^m)
		}
		else {
			c >>= 1;		// c_(m-1) = c_m/2      (aₘ is 0)
		}
		d >>= 2;			// d_(m-1) = d_m/4
	}

	return c;				// c_(-1)

}


__kernel void finda(	__global uint *g_found,
			__global long *g_a,
			const ulong p,
			const ulong maxa){
			
	const uint gid = get_global_id(0);
	const uint gs = get_global_size(0);
	const uint stride = gs*2;
	
	for(long a=1+gid*2; a<=maxa; a+=stride){
		ulong b2 = p-a*a;
		ulong b = isqrt(b2);
		if(b*b == b2){
			uint i = atomic_inc(&g_found[0]);	// found solution
			if(i < ACUBUFFER){
				g_a[i] = (a%4 == 3) ? -a : a;
			}
		}
		if( atomic_or(&g_found[0], 0) ){		// done if any thread finds solution
			break;
		}
	}
}


__kernel void findc(	__global uint *g_found,
			__global long *g_c,
			const ulong p4,
			const ulong maxd){
			
	const uint gid = get_global_id(0);
	const uint gs = get_global_size(0);
	
	for(ulong d=1+gid; d<=maxd; d+=gs){
		ulong c2 = p4 - 27*d*d;
		long c = isqrt(c2);
		if(c*c == c2){
			uint i = atomic_inc(&g_found[0]);	// found solution
			if(i < ACUBUFFER){
				g_c[i] = (c%3 == 2) ? -c : c;
			}
		}
		if( atomic_or(&g_found[0], 0) ){		// done if any thread finds solution
			break;
		}
	}
}


__kernel void findu(	__global uint *g_found,
			__global long *g_u,
			const ulong p4,
			const ulong maxv,
			const uint umod){
			
	const uint gid = get_global_id(0);
	const uint gs = get_global_size(0);
	
	for(ulong v=1+gid; v<=maxv; v+=gs){
		ulong u2 = p4 - 3*v*v;
		long u = isqrt(u2);
		if(u*u == u2){
			uint i = atomic_inc(&g_found[0]);	// found solution
			if(i < ACUBUFFER){
				g_u[i] = (u%3 == umod) ? u : -u;
			}
		}
		if( atomic_or(&g_found[0], 0) ){		// done if any thread finds solution
			break;
		}
	}
}


__kernel void clearacu( __global uint *g_found ){

	const uint gid = get_global_id(0);

	if(gid == 0){
		g_found[0] = 0;
	}
}

			






















