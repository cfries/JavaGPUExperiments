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

/**
 * An example illustrating the behaviour of SIMD versus MIMD on code that contains an if-branch.
 * 
 * The numerical algorithms performs the calculation
 * 	x(i+1) = x(i) + r * x(i) / DeltaT
 * which is a simple Euler-Scheme approximation for x(0) * exp(r T). The calculation is performed is x(0) != 0.
 * 
 * Due to synchronisation (SIMD) in GPUs and due to branch prediction in CPUs the behaviour of the run-time highly depends on the structure of the x(0) vector.
 * The performance characteristics depend on the ordering of the initial value.
 * 
 * @author Christian Fries
 */
public class OpenCLSpeedTest
{
	private final static int localWorkSize = 128;

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

//		final int size = 100000000;		// 100 million
		final int size = 10000;		// 100 million

		int steps = 2000;

		OpenCLSpeedTest testProgramOnGPU = new OpenCLSpeedTest(CL_DEVICE_TYPE_GPU, 0, 128);
		System.out.print("GPU, 128: ");
		testProgramOnGPU.runWithInitialValuesAndRates(i -> 1.0f, i -> 1.0f, size, steps);
		System.out.print("GPU, 128: ");
		testProgramOnGPU.runWithInitialValuesAndRates(i -> 0.0f, i -> 1.0f, size, steps);
		System.out.print("GPU, 128: ");
		testProgramOnGPU.runWithInitialValuesAndRates(i -> i < size/2 ? 0.0f : 1.0f, i -> 1.0f, size, steps);
		System.out.print("GPU, 128: ");
		testProgramOnGPU.runWithInitialValuesAndRates(i -> i % 2 == 0 ? 0.0f : 1.0f, i -> 1.0f, size, steps);
		System.out.print("GPU, 128: ");
		testProgramOnGPU.runWithInitialValuesAndRates(i -> (i/2) % 2 == 0 ? 0.0f : 1.0f, i -> 1.0f, size, steps);
		System.out.print("GPU, 128: ");
		testProgramOnGPU.runWithInitialValuesAndRates(i -> (i/8) % 2 == 0 ? 0.0f : 1.0f, i -> 1.0f, size, steps);
		System.out.print("GPU, 128: ");
		testProgramOnGPU.runWithInitialValuesAndRates(i -> (i/24) % 2 == 0 ? 0.0f : 1.0f, i -> 1.0f, size, steps);

		System.out.println();

		OpenCLSpeedTest testProgramOnCPU = new OpenCLSpeedTest(CL_DEVICE_TYPE_CPU, 0, 128);
		System.out.print("CPU, 128: ");
		testProgramOnCPU.runWithInitialValuesAndRates(i -> 1.0f, i -> 1.0f, size, steps);
		System.out.print("CPU, 128: ");
		testProgramOnCPU.runWithInitialValuesAndRates(i -> 0.0f, i -> 1.0f, size, steps);
		System.out.print("CPU, 128: ");
		testProgramOnCPU.runWithInitialValuesAndRates(i -> i < size/2 ? 0.0f : 1.0f, i -> 1.0f, size, steps);
		System.out.print("CPU, 128: ");
		testProgramOnCPU.runWithInitialValuesAndRates(i -> i % 2 == 0 ? 0.0f : 1.0f, i -> 1.0f, size, steps);
		System.out.print("CPU, 128: ");
		testProgramOnCPU.runWithInitialValuesAndRates(i -> (i/2) % 2 == 0 ? 0.0f : 1.0f, i -> 1.0f, size, steps);
		System.out.print("CPU, 128: ");
		testProgramOnCPU.runWithInitialValuesAndRates(i -> (i/8) % 2 == 0 ? 0.0f : 1.0f, i -> 1.0f, size, steps);
		System.out.print("CPU, 128: ");
		testProgramOnCPU.runWithInitialValuesAndRates(i -> (i/24) % 2 == 0 ? 0.0f : 1.0f, i -> 1.0f, size, steps);

		OpenCLSpeedTest testProgramOnCPU2 = new OpenCLSpeedTest(CL_DEVICE_TYPE_CPU, 0, 1);
		System.out.print("CPU, 128: ");
		testProgramOnCPU2.runWithInitialValuesAndRates(i -> 1.0f, i -> 1.0f, size, steps);
		System.out.print("CPU, 128: ");
		testProgramOnCPU2.runWithInitialValuesAndRates(i -> 0.0f, i -> 1.0f, size, steps);
		System.out.print("CPU, 128: ");
		testProgramOnCPU2.runWithInitialValuesAndRates(i -> i < size/2 ? 0.0f : 1.0f, i -> 1.0f, size, steps);
		System.out.print("CPU, 128: ");
		testProgramOnCPU2.runWithInitialValuesAndRates(i -> i % 2 == 0 ? 0.0f : 1.0f, i -> 1.0f, size, steps);
		System.out.print("CPU, 128: ");
		testProgramOnCPU2.runWithInitialValuesAndRates(i -> (i/2) % 2 == 0 ? 0.0f : 1.0f, i -> 1.0f, size, steps);
		System.out.print("CPU, 128: ");
		testProgramOnCPU2.runWithInitialValuesAndRates(i -> (i/8) % 2 == 0 ? 0.0f : 1.0f, i -> 1.0f, size, steps);
		System.out.print("CPU, 128: ");
		testProgramOnCPU2.runWithInitialValuesAndRates(i -> (i/24) % 2 == 0 ? 0.0f : 1.0f, i -> 1.0f, size, steps);
	}

	/**
	 * Create the test setup. Initializes OpenCL on the given device.
	 * 
	 * @param clDeviceType The device type. May be CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU or CL_DEVICE_TYPE_ALL.
	 * @param deviceIndex The index of the device, in case there are multiple. E.g. on a MacBook Pro 2017 you have two graphic cards: 0 = Intel HD 630, 1 = AMD Radeon Pro 560.
	 * @param localWorkSize The local work size to be used.
	 */
	public OpenCLSpeedTest(final long clDeviceType, final int deviceIndex, final int localWorkSize) {
		super();

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
	private static String programSource1 =
			"__kernel void "+
					"sampleKernel(__global const float *a,"
					+ "             __global const float *b,"
					+ "             __global float *c,"
					+ "             const int steps)"
					+ "{"
					+ "  int gid = get_global_id(0);"
					+ "  float x = a[gid];"
					+ "  if(x != 0.0) {"
					+ "    float r = b[gid];"
					+ "    for(int j=0; j<steps; j++) {"
					+ "      x = x + r * x / steps;"
					+ "    }"
					+ "  }"
					+ "  c[gid] = x;"					
					+ "}";

	/**
	 * The source code of the OpenCL program to execute
	 */
	private static String programSource =
			"__kernel void "+
					"sampleKernel(__global const float *a,"
					+ "             __global const float *b,"
					+ "             __global float *c,"
					+ "             const int length)"
					+ "{"
					+ "  int steps = get_global_id(0);"
					+ "  for(int i=0; i<length; i++) {"
					+ "    float x = a[i];"
					+ "    float r = b[i];"
					+ "    for(int j=0; j<steps; j++) {"
					+ "      x = x + r * x / steps;"
					+ "    }"
					+ "    c[i] = x;"					
					+ "  }"
					+ "}";


	/**
	 * Run the test program.
	 * 
	 * @param initialValue Initial value as a function of the index of the vector.
	 * @param rate Rate as a function of the index of the vector.
	 * @param size Size of the vector to be used. This parameter scales the time required for the calculation.
	 * @param steps Number of approximation steps to be used. This parameter scales the time required for the calculation.
	 */
	private void runWithInitialValuesAndRates(Function<Integer, Float> initialValue, Function<Integer, Float> rate, int size, int steps) {
		// Create input- and output data
		final float srcArrayA[] = new float[size];
		final float srcArrayB[] = new float[size];
		final float dstArray[] = new float[size];
		for (int i=0; i<size; i++)
		{
			srcArrayA[i] = initialValue.apply(i);
			srcArrayB[i] = rate.apply(i);
		}
		final Pointer srcA = Pointer.to(srcArrayA);
		final Pointer srcB = Pointer.to(srcArrayB);
		final Pointer dst = Pointer.to(dstArray);

		// Create the program from the source code
		final cl_program program = clCreateProgramWithSource(context, 1, new String[]{ programSource }, null, null);

		long timeCompileStart = System.currentTimeMillis();

		// Build the program
		clBuildProgram(program, 0, null, null, null, null);

		// Create the kernel
		final cl_kernel kernel = clCreateKernel(program, "sampleKernel", null);

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
		clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[] { steps }));

		// Set the work-item dimensions
		final long global_work_size[] = new long[] { size };
		final long local_work_size[] = null; //new long[] { localWorkSize };

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

		System.out.print(String.format("\t compile %5.3f s", (timeCompileEnd-timeCompileStart)/1000.0));
		System.out.print(String.format("\t   alloc %5.3f s", (timePrepareEnd-timeCompileEnd)/1000.0));
		System.out.print(String.format("\t    calc %5.3f s", (timeCalcEnd-timePrepareEnd)/1000.0));

		// Verify the result
		boolean passed = true;
		final float epsilon = 1e-7f;

		float x = 1.0f;
		float r = 1.0f;
		if(x != 0) {
			for(int j=0; j<steps; j++) {
				x = x + 1.0f * x / steps;
			}
		}

		for (int i=0; i<size; i++) {
			final float y = dstArray[i];
			final boolean epsilonEqual = Math.abs(srcArrayA[i] * x - dstArray[i]) <= epsilon * Math.abs(x);
			if (!epsilonEqual)
			{
				passed = false;
				break;
			}
		}
		System.out.print("\t test "+(passed?"PASSED":"FAILED"));

		double numberOfNonZeroInitialValues = IntStream.range(0, size).mapToDouble(i -> initialValue.apply(i)).filter(u -> u > 0).count();
		System.out.print("\t" + Math.round(numberOfNonZeroInitialValues/size*100) + "%");
		System.out.print("\t");

		for(int i=0; i<48; i++) System.out.printf("%1.0f", initialValue.apply(i));
		System.out.print("...");
		for(int i=0; i<48; i++) System.out.printf("%1.0f", initialValue.apply(size/2+i));

		System.out.println();
	}


	private float[] pureJavaBenchmark(float[] initialValue, float[] rate, int steps) {

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