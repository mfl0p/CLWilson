/* 
	common.cl -- Bryan Little, Yves Gallot, Jul 2025

	Wilson search OpenCL common kernel functions

*/


// using largest local size for reduction kernels
#ifdef __NV_CL_C_VERSION
	#define LSIZE 1024
#else
	#define LSIZE 256
#endif

// r0 + 2^64 * r1 = (a0 + 2^64 * a1) + (b0 + 2^64 * b1)
ulong2 add_wide(const ulong2 a, const ulong2 b)
{
	ulong2 r;

#ifdef __NV_CL_C_VERSION
	const uint a0 = (uint)(a.s0), a1 = (uint)(a.s0 >> 32), a2 = (uint)(a.s1), a3 = (uint)(a.s1 >> 32);
	const uint b0 = (uint)(b.s0), b1 = (uint)(b.s0 >> 32), b2 = (uint)(b.s1), b3 = (uint)(b.s1 >> 32);
	uint c0, c1, c2, c3;

	asm volatile ("add.cc.u32 %0, %1, %2;" : "=r" (c0) : "r" (a0), "r" (b0));
	asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r" (c1) : "r" (a1), "r" (b1));
	asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r" (c2) : "r" (a2), "r" (b2));
	asm volatile ("addc.u32 %0, %1, %2;" : "=r" (c3) : "r" (a3), "r" (b3));

	r.s0 = upsample(c1, c0); r.s1 = upsample(c3, c2);
#else
	r = a + b;
	if (r.s0 < a.s0) r.s1 += 1;
#endif

	return r;
}

// r0 + 2^64 * r1 = a * b
ulong2 mul_wide(const ulong a, const ulong b)
{
	ulong2 r;

#ifdef __NV_CL_C_VERSION
	const uint a0 = (uint)(a), a1 = (uint)(a >> 32);
	const uint b0 = (uint)(b), b1 = (uint)(b >> 32);

	uint c0 = a0 * b0, c1 = mul_hi(a0, b0), c2, c3;

	asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r" (c1) : "r" (a0), "r" (b1), "r" (c1));
	asm volatile ("madc.hi.u32 %0, %1, %2, 0;" : "=r" (c2) : "r" (a0), "r" (b1));

	asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r" (c2) : "r" (a1), "r" (b1), "r" (c2));
	asm volatile ("madc.hi.u32 %0, %1, %2, 0;" : "=r" (c3) : "r" (a1), "r" (b1));

	asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r" (c1) : "r" (a1), "r" (b0), "r" (c1));
	asm volatile ("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r" (c2) : "r" (a1), "r" (b0), "r" (c2));
	asm volatile ("addc.u32 %0, %1, 0;" : "=r" (c3) : "r" (c3));

	r.s0 = upsample(c1, c0); r.s1 = upsample(c3, c2);
#else
	r.s0 = a * b; r.s1 = mul_hi(a, b);
#endif

	return r;
}

// a0 + 2^64 * a1 < b0 + 2^64 * b1
bool is_less_than(const ulong2 a, const ulong2 b)
{
	return (a.s1 < b.s1) || ((a.s1 == b.s1) && (a.s0 < b.s0));
}

// p * p_inv = 1 (mod 2^64) (Newton's method)
ulong invert(const ulong p)
{
	ulong p_inv = 1, prev = 0;
	while (p_inv != prev) { prev = p_inv; p_inv *= 2 - p * p_inv; }
	return p_inv;
}

// r = x + y (mod p) where 0 <= r < p
ulong add_mod(const ulong x, const ulong y, const ulong p)
{
	const ulong cp = (x >= p - y) ? p : 0;
	return x + y - cp;
}

// r = x + y (mod p) where 0 <= r < p, c is the carry
ulong add_mod_c(const ulong x, const ulong y, const ulong p, ulong * c)
{
	const bool carry = (x >= p - y);
	const ulong cp = carry ? p : 0;
	*c = carry ? 1 : 0;
	return x + y - cp;
}

// r = x - y (mod p) where 0 <= r < p
ulong sub_mod(const ulong x, const ulong y, const ulong p)
{
	const ulong cp = (x < y) ? p : 0;
	return x - y + cp;
}

// r = x - y (mod p) where 0 <= r < p, c is the carry
ulong sub_mod_c(const ulong x, const ulong y, const ulong p, ulong * c)
{
	const bool carry = (x < y);
	const ulong cp = carry ? p : 0;
	*c = carry ? 1 : 0;
	return x - y + cp;
}

// "double-precision" variant Montgomery arithmetic. See:
// Peter L. Montgomery, Modular multiplication without trial division, Math. Comp.44 (1985), 519–521.
// Dorais, F. G.; Klyve, D., "A Wieferich Prime Search Up to 6.7x10^15", Journal of Integer Sequences. 14 (9), 2011.

// 2^64 mod p^2 is (2^64, p^2) residue of 1
ulong2 m2p_one(const ulong p)
{
	if ((p >> 32) == 0)
	{
		const ulong p2 = p * p, r_p2 = (-p2) % p2;	// 2^64 mod p^2
		return (ulong2)(r_p2 % p, r_p2 / p);
	}
	// 2^64 mod p^2 = 2^64
	const ulong mp = -p;	// 2^64 - p
	return (ulong2)(mp % p, mp / p + 1);
}

// r0 + p * r1 = 2 * (x0 + p * x1) (mod p^2) where 0 <= r0, r1 < p
ulong2 m2p_dup(const ulong2 x, const ulong p)
{
	ulong c;
	const ulong l = add_mod_c(x.s0, x.s0, p, &c);
	const ulong h = add_mod(x.s1 + c, x.s1, p);
	return (ulong2)(l, h);
}

// r0 + p * r1 = (x0 + p * x1) + (y0 + p * y1) (mod p^2) where 0 <= r0, r1 < p
ulong2 m2p_add(const ulong2 x, const ulong2 y, const ulong p)
{
	ulong c;
	const ulong l = add_mod_c(x.s0, y.s0, p, &c);
	const ulong h = add_mod(x.s1 + c, y.s1, p);
	return (ulong2)(l, h);
}

// r0 + p * r1 = (x0 + p * x1)^2 (mod p^2) where 0 <= r0, r1 < p
ulong2 m2p_square(const ulong2 x, const ulong p, const ulong q)
{
	const ulong2 t = mul_wide(x.s0, x.s0);
	const ulong u0 = q * t.s0;
	const ulong t1 = t.s1;
	const ulong v1 = mul_hi(p, u0);

	const ulong2 x01 = mul_wide(x.s0, x.s1);
	const ulong2 x01u = add_wide(x01, (ulong2)(u0, 0));
	// 0 <= tp < 2p^2: 129 bits
	const ulong2 tp = add_wide(x01u, x01); bool tp_carry = is_less_than(tp, x01);
	// 0 <= tp_h < 2p. tp_h >= p if tp_h >= 2^64 or tp_h >= p
	const ulong tp_h = tp.s1, tpc = (tp_carry | (tp_h >= p)) ? p : 0;
	const ulong up0 = q * tp.s0;
	const ulong t1p = tp_h - tpc;	// 0 <= t1p < p
	const ulong v1p = mul_hi(p, up0);

	// 0 <= t1, v1 < p, 0 <= t1p, v1p < p
	ulong c;
	const ulong z0 = sub_mod_c(t1, v1, p, &c);
	const ulong z1 = sub_mod(t1p, v1p + c, p);
	return (ulong2)(z0, z1);
}

// r0 + p * r1 = (x0 + p * x1) * (y0 + p * y1) (mod p^2) where 0 <= r0, r1 < p
ulong2 m2p_mul(const ulong2 x, const ulong2 y, const ulong p, const ulong q)
{
	const ulong2 t = mul_wide(x.s0, y.s0);
	const ulong u0 = q * t.s0;
	const ulong t1 = t.s1;
	const ulong v1 = mul_hi(p, u0);

	const ulong2 x0y1u = add_wide(mul_wide(x.s0, y.s1), (ulong2)(u0, 0));
	const ulong2 x1y0 = mul_wide(x.s1, y.s0);
	// 0 <= tp < 2p^2: 129 bits
	const ulong2 tp = add_wide(x0y1u, x1y0); bool tp_carry = is_less_than(tp, x1y0);
	// 0 <= tp_h < 2p. tp_h >= p if tp_h >= 2^64 or tp_h >= p
	const ulong tp_h = tp.s1, tpc = (tp_carry | (tp_h >= p)) ? p : 0;
	const ulong up0 = q * tp.s0;
	const ulong t1p = tp_h - tpc;	// 0 <= t1p < p
	const ulong v1p = mul_hi(p, up0);

	// 0 <= t1, v1 < p, 0 <= t1p, v1p < p
	ulong c;
	const ulong z0 = sub_mod_c(t1, v1, p, &c);
	const ulong z1 = sub_mod(t1p, v1p + c, p);
	return (ulong2)(z0, z1);
}

// r0 + p * r1 = x * y (mod p^2) where 0 <= r0, r1 < p
ulong2 m2p_mul_s(const ulong x, const ulong y, const ulong p, const ulong q)
{
	const ulong2 t = mul_wide(x, y);
	const ulong u0 = q * t.s0;
	const ulong t1 = t.s1;
	const ulong v1 = mul_hi(p, u0);

	const ulong v1p = mul_hi(p, q * u0);

	ulong c;
	const ulong z0 = sub_mod_c(t1, v1, p, &c);
	const ulong z1 = sub_mod(0, v1p + c, p);
	return (ulong2)(z0, z1);
}

// r0 + p * r1 = (x0 + p * x1) * (y0 + p * y1) (mod p^2) where 0 <= r0, r1 < p
ulong2 m2p_mul_r2(const ulong x, const ulong2 y, const ulong p, const ulong q)
{
	const ulong2 t = mul_wide(x, y.s0);
	const ulong u0 = q * t.s0;
	const ulong t1 = t.s1;
	const ulong v1 = mul_hi(p, u0);

	const ulong2 x0y1u = add_wide(mul_wide(x, y.s1), (ulong2)(u0, 0));

//	const ulong2 x1y0 = mul_wide(x.s1, y.s0);

	// 0 <= tp < 2p^2: 129 bits
//	const ulong2 tp = add_wide(x0y1u, x1y0); bool tp_carry = is_less_than(tp, x1y0);
	const ulong2 tp = x0y1u; bool tp_carry = false;

	// 0 <= tp_h < 2p. tp_h >= p if tp_h >= 2^64 or tp_h >= p
	const ulong tp_h = tp.s1, tpc = (tp_carry | (tp_h >= p)) ? p : 0;
	const ulong up0 = q * tp.s0;
	const ulong t1p = tp_h - tpc;	// 0 <= t1p < p
	const ulong v1p = mul_hi(p, up0);

	// 0 <= t1, v1 < p, 0 <= t1p, v1p < p
	ulong c;
	const ulong z0 = sub_mod_c(t1, v1, p, &c);
	const ulong z1 = sub_mod(t1p, v1p + c, p);
	return (ulong2)(z0, z1);
}

// To convert a residue to an integer, apply Algorithm REDC
ulong2 m2p_get(const ulong2 x, const ulong p, const ulong q)
{
	const ulong u0 = q * x.s0;
	const ulong v1 = mul_hi(p, u0);

	const ulong tp = x.s1 + u0;
	const ulong up0 = q * tp;
	const ulong t1p = (tp < x.s1) ? 1 : 0;
	const ulong v1p = mul_hi(p, up0);

	ulong c;
	const ulong z0 = sub_mod_c(0, v1, p, &c);
	const ulong z1 = sub_mod(t1p, v1p + c, p);
	return (ulong2)(z0, z1);
}




