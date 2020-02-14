package org.jocl.samples;
/*
 * JOCL - Java bindings for OpenCL
 *
 * Copyright 2009 Marco Hutter - http://www.jocl.org/
 */



import static org.jocl.CL.CL_CONTEXT_PLATFORM;
import static org.jocl.CL.CL_DEVICE_TYPE_ALL;
import static org.jocl.CL.CL_MEM_COPY_HOST_PTR;
import static org.jocl.CL.CL_MEM_READ_ONLY;
import static org.jocl.CL.CL_MEM_READ_WRITE;
import static org.jocl.CL.CL_TRUE;
import static org.jocl.CL.clBuildProgram;
import static org.jocl.CL.clCreateBuffer;
import static org.jocl.CL.clCreateCommandQueue;
import static org.jocl.CL.clCreateContext;
import static org.jocl.CL.clCreateKernel;
import static org.jocl.CL.clCreateProgramWithSource;
import static org.jocl.CL.clEnqueueNDRangeKernel;
import static org.jocl.CL.clEnqueueReadBuffer;
import static org.jocl.CL.clGetDeviceIDs;
import static org.jocl.CL.clGetPlatformIDs;
import static org.jocl.CL.clReleaseCommandQueue;
import static org.jocl.CL.clReleaseContext;
import static org.jocl.CL.clReleaseKernel;
import static org.jocl.CL.clReleaseMemObject;
import static org.jocl.CL.clReleaseProgram;
import static org.jocl.CL.clSetKernelArg;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;

/**
 * A small JOCL sample.
 */
public class JOCLSample
{
	/**
	 * The source code of the OpenCL program to execute
	 */
	private static String programSource =
			"__kernel void "+
					"sampleKernel(__global const float *a,"+
					"             __global const float *b,"+
					"             __global float *c)"+
					"{"+
					"    int gid = get_global_id(0);"+
					"    c[gid] = a[gid] * b[gid];"+
					"}";


	/**
	 * The entry point of this sample
	 *
	 * @param args Not used
	 */
	public static void main(final String args[])
	{
		// Create input- and output data
		final int n = 10;
		final float srcArrayA[] = new float[n];
		final float srcArrayB[] = new float[n];
		final float dstArray[] = new float[n];
		for (int i=0; i<n; i++)
		{
			srcArrayA[i] = i;
			srcArrayB[i] = i;
		}
		final Pointer srcA = Pointer.to(srcArrayA);
		final Pointer srcB = Pointer.to(srcArrayB);
		final Pointer dst = Pointer.to(dstArray);

		// The platform, device type and device number
		// that will be used
		final int platformIndex = 0;
		final long deviceType = CL_DEVICE_TYPE_ALL;
		final int deviceIndex = 0;

		// Enable exceptions and subsequently omit error checks in this sample
		CL.setExceptionsEnabled(true);

		// Obtain the number of platforms
		final int numPlatformsArray[] = new int[1];
		clGetPlatformIDs(0, null, numPlatformsArray);
		final int numPlatforms = numPlatformsArray[0];

		// Obtain a platform ID
		final cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
		clGetPlatformIDs(platforms.length, platforms, null);
		final cl_platform_id platform = platforms[platformIndex];

		// Initialize the context properties
		final cl_context_properties contextProperties = new cl_context_properties();
		contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

		// Obtain the number of devices for the platform
		final int numDevicesArray[] = new int[1];
		clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
		final int numDevices = numDevicesArray[0];

		// Obtain a device ID
		final cl_device_id devices[] = new cl_device_id[numDevices];
		clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
		final cl_device_id device = devices[deviceIndex];

		// Create a context for the selected device
		final cl_context context = clCreateContext(
				contextProperties, 1, new cl_device_id[]{device},
				null, null, null);

		// Create a command-queue for the selected device
		final cl_command_queue commandQueue =
				clCreateCommandQueue(context, device, 0, null);

		// Allocate the memory objects for the input- and output data
		final cl_mem memObjects[] = new cl_mem[3];
		memObjects[0] = clCreateBuffer(context,
				CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				Sizeof.cl_float * n, srcA, null);
		memObjects[1] = clCreateBuffer(context,
				CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				Sizeof.cl_float * n, srcB, null);
		memObjects[2] = clCreateBuffer(context,
				CL_MEM_READ_WRITE,
				Sizeof.cl_float * n, null, null);

		// Create the program from the source code
		final cl_program program = clCreateProgramWithSource(context,
				1, new String[]{ programSource }, null, null);

		// Build the program
		clBuildProgram(program, 0, null, null, null, null);

		// Create the kernel
		final cl_kernel kernel = clCreateKernel(program, "sampleKernel", null);

		// Set the arguments for the kernel
		clSetKernelArg(kernel, 0,
				Sizeof.cl_mem, Pointer.to(memObjects[0]));
		clSetKernelArg(kernel, 1,
				Sizeof.cl_mem, Pointer.to(memObjects[1]));
		clSetKernelArg(kernel, 2,
				Sizeof.cl_mem, Pointer.to(memObjects[2]));

		// Set the work-item dimensions
		final long global_work_size[] = new long[]{n};
		final long local_work_size[] = new long[]{1};

		// Execute the kernel
		clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
				global_work_size, local_work_size, 0, null, null);

		// Read the output data
		clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0,
				n * Sizeof.cl_float, dst, 0, null, null);

		// Release kernel, program, and memory objects
		clReleaseMemObject(memObjects[0]);
		clReleaseMemObject(memObjects[1]);
		clReleaseMemObject(memObjects[2]);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(commandQueue);
		clReleaseContext(context);

		// Verify the result
		boolean passed = true;
		final float epsilon = 1e-7f;
		for (int i=0; i<n; i++)
		{
			final float x = dstArray[i];
			final float y = srcArrayA[i] * srcArrayB[i];
			final boolean epsilonEqual = Math.abs(x - y) <= epsilon * Math.abs(x);
			if (!epsilonEqual)
			{
				passed = false;
				break;
			}
		}
		System.out.println("Test "+(passed?"PASSED":"FAILED"));
		if (n <= 10)
		{
			System.out.println("Result: "+java.util.Arrays.toString(dstArray));
		}
	}
}