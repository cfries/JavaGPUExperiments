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

import com.christianfries.opencl.examples.OpenCLSpeedTest.Method;

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
	public enum Method {
		JAVA,				// Use native Java implementation
		OPEN_CL_CPU,		// Use OpenCL implementation on CPU
		OPEN_CL_GPU,		// Use OpenCL implementation on GPU (uses the GPU with the highest device index)
		OPEN_CL_GPU_0,		// Use OpenCL implementation on GPU
		OPEN_CL_GPU_1		// Use OpenCL implementation on GPU
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

		final int size = 102400000;		// 102.4 million

		int steps;

		steps = 200;

		List<Function<Integer, Float>> initialValues = List.of(
				i -> 1.0f,									// Initial value constant 1.: 111111111111111111111111111111111111111111111111...111111111111111111111111111111111111111111111111
				i -> 0.0f,									// Initial value constant 0.: 000000000000000000000000000000000000000000000000...000000000000000000000000000000000000000000000000
				i -> i < size/2 ? 0.0f : 1.0f,				// Initial value 50% 0 and 1: 000000000000000000000000000000000000000000000000...111111111111111111111111111111111111111111111111 
				i -> i % 2 == 0 ? 0.0f : 1.0f,				// Initial value 50% 0 and 1: 010101010101010101010101010101010101010101010101...010101010101010101010101010101010101010101010101
				i -> (i/2) % 2 == 0 ? 0.0f : 1.0f,			// Initial value 50% 0 and 1: 001100110011001100110011001100110011001100110011...001100110011001100110011001100110011001100110011
				i -> (i/8) % 2 == 0 ? 0.0f : 1.0f,			// Initial value 50% 0 and 1: 000000001111111100000000111111110000000011111111...000000001111111100000000111111110000000011111111
				i -> (i/16) % 2 == 0 ? 0.0f : 1.0f,			// Initial value 50% 0 and 1: 000000000000000000000000111111111111111111111111...111111111111111100000000000000000000000011111111
				i -> (i/1024) % 2 == 0 ? 0.0f : 1.0f		// Initial value 50% 0 and 1: 000000000000000000000000111111111111111111111111...111111111111111100000000000000000000000011111111
		);

		/*
		 * Java code
		 */
		System.out.println("Java:");
		OpenCLSpeedTest testProgramJava = new OpenCLSpeedTest(Method.JAVA);
		for(Function<Integer, Float> initialValue : initialValues) {
			testProgramJava.runWithInitialValuesAndRates(initialValue, i -> 1.0f, size, steps);
		}
		testProgramJava.cleanUp();

		System.out.println();

		steps = 1000;

		/*
		 * OpenCL with CPU
		 */
		System.out.println("OpenCL on CPU:");
		OpenCLSpeedTest testProgramOnCPU = new OpenCLSpeedTest(Method.OPEN_CL_CPU);
		for(Function<Integer, Float> initialValue : initialValues) {
			testProgramOnCPU.runWithInitialValuesAndRates(initialValue, i -> 1.0f, size, steps);
		}
		testProgramOnCPU.cleanUp();

		System.out.println();

		steps = 20000;

		/*
		 * OpenCL with GPU
		 */
//		System.out.println("OpenCL on GPU (Intel HD 630):");
//		OpenCLSpeedTest testProgramOnGPU = new OpenCLSpeedTest(Method.OPEN_CL_GPU_0);
//		for(Function<Integer, Float> initialValue : initialValues) {
//			testProgramOnGPU.runWithInitialValuesAndRates(initialValue, i -> 1.0f, size, steps);
//		}
//		testProgramOnGPU.cleanUp();
//
//		System.out.println();

		/*
		 * OpenCL with GPU
		 */
		System.out.println("OpenCL on GPU (AMD Radeon Pro 560):");
		OpenCLSpeedTest testProgramOnGPU1 = new OpenCLSpeedTest(Method.OPEN_CL_GPU);
		for(Function<Integer, Float> initialValue : initialValues) {
			testProgramOnGPU1.runWithInitialValuesAndRates(initialValue, i -> 1.0f, size, steps);
		}
		testProgramOnGPU1.cleanUp();

		System.out.println();
	}

	/**
	 * Create the test setup. Initializes OpenCL on the given device.
	 * 
	 * @param method Specify which platform / device we use (Java, OpenCL CPU, OpenCL GPU)
	 */
	public OpenCLSpeedTest(final Method method) {
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
		case OPEN_CL_GPU:
			clDeviceType = CL_DEVICE_TYPE_GPU;
			deviceIndex = -1;
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
		final int clDeviceIndex = deviceIndex >= 0 ? deviceIndex : (devices.length + deviceIndex);
		device = devices[clDeviceIndex];

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
					"evolve(__global const float *a,"
					+ "     __global const float *b,"
					+ "     __global float *c,"
					+ "     const int steps)"
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
		final float dstArray[];
		for (int i=0; i<size; i++)
		{
			srcArrayA[i] = initialValue.apply(i);
			srcArrayB[i] = rate.apply(i);
		}

		if(method.equals(Method.JAVA)) {

			long timeCalcStart = System.currentTimeMillis();

			dstArray = pureJavaBenchmark(srcArrayA, srcArrayB, steps);

			long timeCalcEnd = System.currentTimeMillis();

			System.out.print(String.format(" %7d steps ", steps));
			System.out.print(String.format("\t compile: %5s  ", "---"));
			System.out.print(String.format("  alloc: %5s  ", "---"));
			System.out.print(String.format("  calc: %5.2f s", (timeCalcEnd-timeCalcStart)/1000.0));
			System.out.print(String.format(" (%6.2f ms / step)", (double)(timeCalcEnd-timeCalcStart)/steps));
		}
		else {
			dstArray = new float[size];
			final Pointer srcA = Pointer.to(srcArrayA);
			final Pointer srcB = Pointer.to(srcArrayB);
			final Pointer dst = Pointer.to(dstArray);

			// Create the program from the source code
			final cl_program program = clCreateProgramWithSource(context, 1, new String[]{ programSource }, null, null);

			long timeCompileStart = System.currentTimeMillis();

			// Build the program
			clBuildProgram(program, 0, null, null, null, null);

			// Create the kernel
			final cl_kernel kernel = clCreateKernel(program, "evolve", null);

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

			System.out.print(String.format(" %7d steps ", steps));
			System.out.print(String.format("\t compile: %5.2f s", (timeCompileEnd-timeCompileStart)/1000.0));
			System.out.print(String.format("  alloc: %5.2f s", (timePrepareEnd-timeCompileEnd)/1000.0));
			System.out.print(String.format("  calc: %5.2f s", (timeCalcEnd-timePrepareEnd)/1000.0));
			System.out.print(String.format(" (%6.2f ms / step)", (double)(timeCalcEnd-timePrepareEnd)/steps));
		}

		// Verify the result
//		boolean passed = true;
//		final float epsilon = 1e-7f;
//
//		float x = 1.0f;
//		float r = 1.0f;
//		if(x != 0) {
//			for(int j=0; j<steps; j++) {
//				x = x + 1.0f * x / steps;
//			}
//		}
//
//		for (int i=0; i<size; i++) {
//			final float y = dstArray[i];
//			final boolean epsilonEqual = Math.abs(srcArrayA[i] * x - dstArray[i]) <= epsilon * Math.abs(x);
//			if (!epsilonEqual)
//			{
//				passed = false;
//				break;
//			}
//		}
//		System.out.print("\t test "+(passed?"PASSED":"FAILED"));

		double numberOfNonZeroInitialValues = IntStream.range(0, size).mapToDouble(i -> initialValue.apply(i)).filter(u -> u > 0).count();
		System.out.print("\t" + Math.round(numberOfNonZeroInitialValues/size*100) + "%");
		System.out.print("\t");

		for(int i=0; i<24; i++) System.out.printf("%1.0f", initialValue.apply(i));
		System.out.print("...");
		for(int i=0; i<24; i++) System.out.printf("%1.0f", initialValue.apply(1024 + i));
		System.out.print("...");
		for(int i=0; i<24; i++) System.out.printf("%1.0f", initialValue.apply(size/2+i));
		System.out.print("...");
		for(int i=0; i<24; i++) System.out.printf("%1.0f", initialValue.apply(size/2 + 1024 +i));

		System.out.println();
	}


	private float[] pureJavaBenchmark(float[] initialValue, float[] rate, int steps) {

		float[] result = new float[initialValue.length];

		IntStream.range(0, initialValue.length).parallel().forEach(i ->
		//		for (int i=0; i<initialValue.length; i++)
		{
			float x = initialValue[i];
			float r = rate[i];
			if(x != 0) {
				for(int j=0; j<steps; j++) {
					x = x + r * x / steps;
				}
			}
			result[i] = x;
		});

		return result;
	}
}