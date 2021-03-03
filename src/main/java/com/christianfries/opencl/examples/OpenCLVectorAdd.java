/*
 * (c) Copyright Christian P. Fries, Germany. Contact: email@christian-fries.de.
 *
 * Created on 14.02.2021
 */

package com.christianfries.opencl.examples;

import static org.jocl.CL.CL_CONTEXT_PLATFORM;
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

import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.IntStream;

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

import com.christianfries.opencl.examples.OpenCLVectorAdd.Method;

/**
 * An example illustrating adding two vectors using OpenCL
 * 	c[i] = a[i] + b[i]
 * 
 * @author Christian Fries
 */
public class OpenCLVectorAdd
{
	public enum Method {
		OPEN_CL_CPU,		// Use OpenCL implementation on CPU
		OPEN_CL_GPU_0,		// Use OpenCL implementation on CPU
		OPEN_CL_GPU_1		// Use OpenCL implementation on CPU
	}

	final Method method;
	final cl_device_id device;
	final cl_context context;
	final cl_command_queue commandQueue;

	/**
	 * The entry point of this sample
	 *
	 * @param args Not used
	 */
	public static void main(final String args[])
	{
		System.out.println("Warning: The program may lead to issues (crash) in case you GPU is busy doing other stuff, e.g. driving a large external monitor.");
		System.out.println();

		final int size = 100000000;		// 100 million
		
		float[] a = new float[size];
		float[] b = new float[size];
		
		Random random = new Random(3141);
		for(int i = 0; i<size; i++) {
			a[i] = random.nextFloat();
			b[i] = random.nextFloat();
		}

		OpenCLVectorAdd openCLVectorAdd = new OpenCLVectorAdd(Method.OPEN_CL_GPU_0);

		float[] c = openCLVectorAdd.add(a, b);

		openCLVectorAdd.cleanUp();

		boolean failed = false;
		for(int i = 0; i<a.length; i++) {
			failed |= (c[i] != a[i] + b[i]);
		}
		System.out.println("Test: " + (failed ? "FAILED" : "PASSED"));
	}

	/**
	 * Create the test setup. Initializes OpenCL on the given device.
	 * 
	 * @param method Specify which platform / device we use (Java, OpenCL CPU, OpenCL GPU)
	 */
	public OpenCLVectorAdd(final Method method) {
		super();
		this.method = method;

		final long clDeviceType;
		final int deviceIndex;

		switch(method) {
		case OPEN_CL_CPU:
		default:
			clDeviceType = CL_DEVICE_TYPE_CPU;
			deviceIndex = 0;
			break;
		case OPEN_CL_GPU_0:
			clDeviceType = CL_DEVICE_TYPE_GPU;
			deviceIndex = 0;
			break;
		case OPEN_CL_GPU_1:
			clDeviceType = CL_DEVICE_TYPE_GPU;
			deviceIndex = 1;
			break;
		}

		/*
		 * Initialize OpenCL (for method=JAVA this is not needed, we do it anyway).
		 */

		// The platform, device type and device number
		// that will be used
		final int platformIndex = 0;

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
		clGetDeviceIDs(platform, clDeviceType, 0, null, numDevicesArray);
		final int numDevices = numDevicesArray[0];

		// Obtain a device ID
		final cl_device_id devices[] = new cl_device_id[numDevices];
		clGetDeviceIDs(platform, clDeviceType, numDevices, devices, null);
		device = devices[deviceIndex];

		// Create a context for the selected device
		context = clCreateContext(contextProperties, 1, new cl_device_id[]{ device }, null, null, null);

		// Create a command-queue for the selected device
		commandQueue = clCreateCommandQueue(context, device, 0, null);
	}

	private void cleanUp() {
		clReleaseCommandQueue(commandQueue);
		clReleaseContext(context);
	}

	/**
	 * The source code of the OpenCL program to execute
	 */
	private static String programSource =
			"__kernel void "+
					"add(__global const float *a,"
					+ "  __global const float *b,"
					+ "  __global float *c)"
					+ "{"
					+ "  int i = get_global_id(0);"
					+ ""
					+ "  c[i] = a[i] + b[i];"
					+ "}";

	/**
	 * Run the test program.
	 * 
	 * @param initialValue Initial value as a function of the index of the vector.
	 * @param rate Rate as a function of the index of the vector.
	 * @param size Size of the vector to be used. This parameter scales the time required for the calculation.
	 * @param steps Number of approximation steps to be used. This parameter scales the time required for the calculation.
	 */
	private float[] add(float[] arrayA, float[] arrayB) {

		int size = arrayA.length;
		float[] result = new float[size];

		final Pointer srcA = Pointer.to(arrayA);
		final Pointer srcB = Pointer.to(arrayB);
		final Pointer dst = Pointer.to(result);

		// Create the program from the source code
		final cl_program program = clCreateProgramWithSource(context, 1, new String[]{ programSource }, null, null);

		long timeCompileStart = System.currentTimeMillis();

		// Build the program
		clBuildProgram(program, 0, null, null, null, null);

		// Create the kernel
		final cl_kernel kernel = clCreateKernel(program, "add", null);

		long timeCompileEnd = System.currentTimeMillis();

		// Allocate the memory objects for the input- and output data
		final cl_mem memObjects[] = new cl_mem[3];
		memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * size, srcA, null);
		memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * size, srcB, null);
		memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, Sizeof.cl_float * size, null, null);

		// Set the arguments for the kernel
		clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(memObjects[0]));
		clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(memObjects[1]));
		clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(memObjects[2]));

		// Set the work-item dimensions
		final long global_work_size[] = new long[] { size };
		final long local_work_size[] = null;	// if you do not specify, OpenCL will try to choose optimal value

		long timePrepareEnd = System.currentTimeMillis();

		// Execute the kernel
		clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, global_work_size, local_work_size, 0, null, null);

		// Read the output data
		clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0, size * Sizeof.cl_float, dst, 0, null, null);

		long timeCalcEnd = System.currentTimeMillis();

		// Release kernel, program, and memory objects
		clReleaseMemObject(memObjects[0]);
		clReleaseMemObject(memObjects[1]);
		clReleaseMemObject(memObjects[2]);

		clReleaseKernel(kernel);
		clReleaseProgram(program);
		
		return result;
	}
}
