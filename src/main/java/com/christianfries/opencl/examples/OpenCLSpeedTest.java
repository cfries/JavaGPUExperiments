package com.christianfries.opencl.examples;

import static org.jocl.CL.CL_CONTEXT_PLATFORM;
import static org.jocl.CL.CL_DEVICE_TYPE_ALL;
import static org.jocl.CL.CL_DEVICE_TYPE_CPU;
import static org.jocl.CL.CL_DEVICE_TYPE_GPU;
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
 * An example illustrating the behaviour of SIMD versus MIMD upon branching.
 */
public class OpenCLSpeedTest
{
	private final static int localWorkSize = 128;

	/**
	 * The source code of the OpenCL program to execute
	 */
	private static String programSource =
			"__kernel void "+
					"sampleKernel(__global const float *a,"+
					"             __global const float *b,"+
					"             __global float *c)"
					+ "{"
					+ "  int gid = get_global_id(0);"
					+ "  int size = get_global_size(0);"
					+ "  float x = a[gid];"
					+ "  int steps = 1000;"					
					+ "  if(x != 0.0) {"
					+ "    float r = b[gid];"
					+ "    for(int j=0; j<steps; j++) {"
					+ "      x = x + r * x / steps;"
					+ "    }"
					+ "  }"
					+ "  c[gid] = x;"					
					+ "}";


	/**
	 * The entry point of this sample
	 *
	 * @param args Not used
	 */
	public static void main(final String args[])
	{
		// Create input- and output data
		final int n = 100000000;		// 100 million
		final float srcArrayA[] = new float[n];
		final float srcArrayB[] = new float[n];
		final float dstArray[] = new float[n];
		for (int i=0; i<n; i++)
		{
			srcArrayA[i] = 1;//i % 2 == 0 ? 0 : 1;
			srcArrayB[i] = 1;
		}
		final Pointer srcA = Pointer.to(srcArrayA);
		final Pointer srcB = Pointer.to(srcArrayB);
		final Pointer dst = Pointer.to(dstArray);

		// The platform, device type and device number
		// that will be used
		final int platformIndex = 0;
		//		final long deviceType = CL_DEVICE_TYPE_ALL;
		//		final long deviceType = CL_DEVICE_TYPE_CPU;
		final long deviceType = CL_DEVICE_TYPE_GPU;
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
		final cl_context context = clCreateContext(contextProperties, 1, new cl_device_id[]{ device }, null, null, null);

		// Create a command-queue for the selected device
		final cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, null);

		// Allocate the memory objects for the input- and output data
		final cl_mem memObjects[] = new cl_mem[3];
		memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * n, srcA, null);
		memObjects[1] = clCreateBuffer(context,
				CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				Sizeof.cl_float * n, srcB, null);
		memObjects[2] = clCreateBuffer(context,
				CL_MEM_READ_WRITE,
				Sizeof.cl_float * n, null, null);

		// Create the program from the source code
		final cl_program program = clCreateProgramWithSource(context,
				1, new String[]{ programSource }, null, null);

		long timeCompileStart = System.currentTimeMillis();

		// Build the program
		clBuildProgram(program, 0, null, null, null, null);

		// Create the kernel
		final cl_kernel kernel = clCreateKernel(program, "sampleKernel", null);

		long timeCompileEnd = System.currentTimeMillis();

		// Set the arguments for the kernel
		clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(memObjects[0]));
		clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(memObjects[1]));
		clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(memObjects[2]));

		// Set the work-item dimensions
		final long global_work_size[] = new long[] { n };
		final long local_work_size[] = new long[] { localWorkSize };

		long timePrepareEnd = System.currentTimeMillis();

		// Execute the kernel
		clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, global_work_size, local_work_size, 0, null, null);

		// Read the output data
		clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0, n * Sizeof.cl_float, dst, 0, null, null);

		long timeCalcEnd = System.currentTimeMillis();

		// Release kernel, program, and memory objects
		clReleaseMemObject(memObjects[0]);
		clReleaseMemObject(memObjects[1]);
		clReleaseMemObject(memObjects[2]);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(commandQueue);
		clReleaseContext(context);

		System.out.println((timeCompileEnd-timeCompileStart)/1000.0);
		System.out.println((timePrepareEnd-timeCompileEnd)/1000.0);
		System.out.println((timeCalcEnd-timePrepareEnd)/1000.0);

		// Verify the result
		boolean passed = true;
		final float epsilon = 1e-7f;

		int steps = 1000;
		float x = 1.0f;
		float r = 1.0f;
		if(x != 0) {
			for(int j=0; j<steps; j++) {
				x = x + 1.0f * x / steps;
			}
		}

		for (int i=0; i<n; i++) {
			final float y = dstArray[i];
			final boolean epsilonEqual = Math.abs(x - dstArray[i]) <= epsilon * Math.abs(x);
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


	float[] pureJavaBenchmark(float[] initialValue, float[] rate, int steps) {

		float[] result = new float[initialValue.length];

		long timeJavaStart = System.currentTimeMillis();

		for (int i=0; i<initialValue.length; i++)
		{
			float x = initialValue[i];
			float r = rate[i];
			if(x != 0) {
				for(int j=0; j<steps; j++) {
					x = x + 1.0f * x / steps;
				}
			}
			result[i] = x;
		}

		long timeJavaEnd = System.currentTimeMillis();

		System.out.println((timeJavaEnd-timeJavaStart)/1000.0);

		return result;
	}
}