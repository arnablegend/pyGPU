import pyopencl as cl
from pyopencl import array
import numpy as np

device = None
ctx = None
queue = None
mf = None
prg_add = None
prg_div = None
prg_sub = None
prg_mul = None
prg_exp = None
prg_log = None
prg_sqrt = None
prg_tanh = None
prg_dot= None
prg_sign= None

def initializeGPU(dev):
	global device, ctx, queue, mf, prg_add, prg_div, prg_sub, prg_mul, prg_exp, prg_log, prg_log, prg_tanh,prg_dot,prg_sign

	device=dev
	ctx=cl.Context(device) #assign a context for a device
	queue=cl.CommandQueue(ctx) #Create a queue for that context. every context has a single queue and work to be done is enqueued
	mf=cl.mem_flags
	prg_add=cl.Program(ctx,"""
		__kernel void sum(__global const float *a_g, __global const float *b_g, __global float *res_g){
				int gid = get_global_id(0);
				res_g[gid] = a_g[gid] + b_g[gid];
		}
		""").build()
	prg_div=cl.Program(ctx,"""
		__kernel void divide(__global const float *a_g, __global const float *b_g, __global float *res_g){
				int gid = get_global_id(0);
				res_g[gid] = a_g[gid] / b_g[gid];
		}
		""").build()
	prg_sub=cl.Program(ctx,"""
		__kernel void sub(__global const float *a_g, __global const float *b_g, __global float *res_g){
				int gid = get_global_id(0);
				res_g[gid] = a_g[gid] - b_g[gid];
		}
		""").build()
	prg_mul=cl.Program(ctx,"""
		__kernel void mul(__global const float *a_g, __global const float *b_g, __global float *res_g){
				int gid = get_global_id(0);
				res_g[gid] = a_g[gid] * b_g[gid];
		}
		""").build()
	prg_exp = cl.Program(ctx,"""
		__kernel void expon(__global const float *a_g,__global float *res_g){
			int gid=get_global_id(0);
			res_g[gid]=exp(a_g[gid]);
		}
		""").build()
	prg_log = cl.Program(ctx,"""
		__kernel void logarithm(__global const float *a_g,__global float *res_g){
			int gid=get_global_id(0);
			res_g[gid]=log(a_g[gid]);
			}
		""").build()
	prg_sqrt = cl.Program(ctx,"""
		__kernel void root(__global const float *a_g,__global float *res_g){
			int gid=get_global_id(0);
			res_g[gid]=sqrt(a_g[gid]);
		}
		""").build()
 	prg_tanh = cl.Program(ctx,"""
		__kernel void tanhyper(__global const float *a_g,__global float *res_g){
			int gid=get_global_id(0);
			res_g[gid]=tanh(a_g[gid]);
		}
		""").build()
	prg_dot = cl.Program(ctx, """
		__kernel void dotprod(__global float* res_g, __global const float* a_g, __global const float* b_g, int wA, int wB){
   			int tx = get_global_id(0); 
   			int ty = get_global_id(1);
   			float value = 0;
   			for (int k = 0; k < wA; ++k){
      			float elementA = a_g[ty * wA + k];
      			float elementB = b_g[k * wB + tx];
      			value += elementA * elementB;
   			}
   			res_g[ty * wA + tx] = value;
		}""").build()
	prg_sign = cl.Program(ctx, """
		__kernel void sign(__global const float *a_g, __global float *res_g) {
  			int gid = get_global_id(0);
  			int sign;
  			if(a_g[gid]>0){
    			sign=1;
  			}else if(a_g[gid]<0){
				sign=-1;
  			}else{
   				sign=0;
  			}
  			res_g[gid]=sign;
		}
		""").build()


def add(arg1,arg2):
	global device, ctx, queue, mf, prg_add

	arr_a=np.array(arg1).astype(np.float32)
	arr_b=np.array(arg2).astype(np.float32)
	res=np.zeros_like(arr_a)
	a_g=cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr_a)
	b_g=cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr_b)
	res_g=cl.Buffer(ctx,mf.WRITE_ONLY,arr_a.nbytes)

	prg_add.sum(queue,arr_a.shape,None,a_g,b_g,res_g)  #deploy the kernel...this is the execution of the program

	cl.enqueue_copy(queue,res,res_g)  #copy from device memory to host machine

	return res 
	
def divide(arg1,arg2):
	global device, ctx, queue, mf, prg_div

	arr_a=np.array(arg1).astype(np.float32)
	arr_b=np.array(arg2).astype(np.float32)
	res=np.zeros_like(arr_a)
	a_g=cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr_a)
	b_g=cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr_b)
	res_g=cl.Buffer(ctx,mf.WRITE_ONLY,arr_a.nbytes)

	prg_div.divide(queue,arr_a.shape,None,a_g,b_g,res_g)

	cl.enqueue_copy(queue,res,res_g)

	return res
	
def subtract(arg1,arg2):
	global device, ctx, queue, mf, prg_sub

	arr_a=np.array(arg1).astype(np.float32)
	arr_b=np.array(arg2).astype(np.float32)
	res=np.zeros_like(arr_a)
	a_g=cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr_a)
	b_g=cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr_b)
	res_g=cl.Buffer(ctx,mf.WRITE_ONLY,arr_a.nbytes)

	prg_sub.sub(queue,arr_a.shape,None,a_g,b_g,res_g)  

	cl.enqueue_copy(queue,res,res_g) 

	return res

def multiply(arg1,arg2):
	global device, ctx, queue, mf, prg_mul

	arr_a=np.array(arg1).astype(np.float32)
	arr_b=np.array(arg2).astype(np.float32)
	res=np.zeros_like(arr_a)
	a_g=cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr_a)
	b_g=cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr_b)
	res_g=cl.Buffer(ctx,mf.WRITE_ONLY,arr_a.nbytes)

	prg_mul.mul(queue,arr_a.shape,None,a_g,b_g,res_g)  

	cl.enqueue_copy(queue,res,res_g) 

	return res

def exp(arg1):
	global device, ctx, queue, mf, prg_exp

	arr_a=np.array(arg1).astype(np.float32)	
	a_g=cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr_a)
	res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)	
	prg_exp.expon(queue, a_np.shape, None, a_g, res_g)
		
	cl.enqueue_copy(queue,res,res_g)

	return res

def log(arg1):
	global device, ctx, queue, mf, prg_log

	arr_a=np.array(arg1).astype(np.float32)
	a_g=cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr_a)
	res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)	
	prg_log.logarithm(queue, a_np.shape, None, a_g, res_g)
		
	cl.enqueue_copy(queue,res,res_g)

	return res

def square(arg1):
	return multiply(arg1,arg1)

def sqrt(arg1):
	global device, ctx, queue, mf, prg_sqrt

	arr_a=fabs(arg1)
	a_g=cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr_a)
	res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)	
	prg_sqrt.root(queue, a_np.shape, None, a_g, res_g)
		
	cl.enqueue_copy(queue,res,res_g)

	return res

def tanh(arg1):
	global device, ctx, queue, mf, prg_tanh

	arr_a=np.array(arg1).astype(np.float32)
	a_g=cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr_a)
	res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)	
	prg_tanh.tanhyper(queue, a_np.shape, None, a_g, res_g)
		
	cl.enqueue_copy(queue,res,res_g)

	return res

def dot(arg1,arg2):
	global device,ctx,queue,mf,prg_dot
	
	arr_a=np.array(arg1).astype(np.float32)
	arr_b=np.array(arg2).astype(np.float32)
	res=np.zeros((arr_a.shape[0],arr_a.shape[0]),dtype=np.float32)
	dim=np.array([arr_a.shape[0]]).astype(np.int32)

	a_g=cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr_a)
	b_g=cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr_b)
	res_g=cl.Buffer(ctx,mf.WRITE_ONLY,res.nbytes)	

	prg_dot.dotprod(queue,(arr_a.shape[0],arr_a.shape[0]),None,res_g,a_g,b_g,dim[0],dim[0])

	cl.enqueue_copy(queue,res,res_g)

	return res

def sign(arg1):
	global device,ctx,queue,mf,prg_sign
	
	arr_a=np.array(arg1).astype(np.float32)	
	a_g=cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr_a)
	res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)	
	prg_sign.sign(queue, a_np.shape, None, a_g, res_g)
		
	cl.enqueue_copy(queue,res,res_g)

	return res
