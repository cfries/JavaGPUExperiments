/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christian-fries.de.
 *
 * Created on 24.02.2017
 */

package com.christianfries.cuda.examples;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;

import jcuda.LogLevel;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;

/**
 * The class RandomVariable represents a random variable being the evaluation of a stochastic process
 * at a certain time within a Monte-Carlo simulation.
 * 
 * It is thus essentially a vector of floating point numbers - the realizations.
 *
 * Accesses performed exclusively through the interface
 * <code>RandomVariableSimpleInterface</code>
 * (and does not mutate the class).
 *
 * 
 * @author Christian Fries
 * @version 1.8
 */
public class RandomVariableSimpleCuda implements RandomVariableSimpleInterface {

	// Static device stuff
	public final static CUdevice device;
	public final static CUcontext context;

	private final static CUfunction add;
	private final static CUfunction div;

	// Initalize cuda
	static {
		// Enable exceptions and omit all subsequent error checks
		JCudaDriver.setExceptionsEnabled(true);
		JCudaDriver.setLogLevel(LogLevel.LOG_DEBUG);

		// Create the PTX file by calling the NVCC
		String ptxFileName = null;
		try {
			ptxFileName = preparePtxFile("RandomVariableSimpleCudaKernel.cu");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		// Initialize the driver and create a context for the first device.
		cuInit(0);
		device = new CUdevice();
		cuDeviceGet(device, 0);
		context = new CUcontext();
		cuCtxCreate(context, 0, device);

		// Load the ptx file.
		CUmodule module = new CUmodule();
		cuModuleLoad(module, ptxFileName);

		add = new CUfunction();
		cuModuleGetFunction(add, module, "cuAdd");
		div = new CUfunction();
		cuModuleGetFunction(div, module, "cuDiv");
	}

	// Need to ref to data here
	private CUdeviceptr realizations;
	private long size;

	/**
	 * Create a stochastic random variable.
	 *
	 * @param realisations the vector of realizations.
	 */
	public RandomVariableSimpleCuda(float[] realisations) {
		super();
		this.realizations = createCUdeviceptr(realisations);
		this.size = realisations.length;
	}

	public RandomVariableSimpleCuda(CUdeviceptr realizations, long size) {
		this.realizations = realizations;
		this.size = size;
	}

	private CUdeviceptr createCUdeviceptr(long size) {
		CUdeviceptr cuDevicePtr = getCUdeviceptr(size);
		return cuDevicePtr;
	}

	public static CUdeviceptr getCUdeviceptr(long size) {
		CUdeviceptr cuDevicePtr = new CUdeviceptr();
		int succ = JCudaDriver.cuMemAlloc(cuDevicePtr, size * Sizeof.FLOAT);
		if(succ != 0) {
			cuDevicePtr = null;
			throw new RuntimeException("Failed creating device vector "+ cuDevicePtr + " with size=" + size);
		}

		return cuDevicePtr;
	}

	/**
	 * Create a vector on device and copy host vector to it.
	 * 
	 * @param values Host vector.
	 * @return Pointer to device vector.
	 */
	private CUdeviceptr createCUdeviceptr(float[] values) {
		CUdeviceptr cuDevicePtr = createCUdeviceptr((long)values.length);
		JCudaDriver.cuMemcpyHtoD(cuDevicePtr, Pointer.to(values),
				(long)values.length * Sizeof.FLOAT);
		return cuDevicePtr;
	}

	@Override
	protected void finalize() throws Throwable {
		System.out.println("Finalizing " + realizations);
		if(realizations != null) JCudaDriver.cuMemFree(realizations);
		super.finalize();
	}


	@Override
	public long size() {
		return size;
	}

	@Override
	public float[] getRealizations() {
		float[] result = new float[(int)size];
		cuMemcpyDtoH(Pointer.to(result), realizations, size * Sizeof.FLOAT);
		return result;
	}

	@Override
	public RandomVariableSimpleInterface add(RandomVariableSimpleInterface randomVariable) {
		CUdeviceptr result = callCudaFunction(add, new Pointer[] {
				Pointer.to(new int[] { (int)size() }),
				Pointer.to(realizations),
				Pointer.to(((RandomVariableSimpleCuda)randomVariable).realizations),
				new Pointer()}
				);

		return new RandomVariableSimpleCuda(result, size());
	}

	@Override
	public RandomVariableSimpleInterface div(RandomVariableSimpleInterface randomVariable) {
		CUdeviceptr result = callCudaFunction(div, new Pointer[] {
				Pointer.to(new int[] { (int)size() }),
				Pointer.to(realizations),
				Pointer.to(((RandomVariableSimpleCuda)randomVariable).realizations),
				new Pointer()}
				);

		return new RandomVariableSimpleCuda(result, size());
	}

	private CUdeviceptr callCudaFunction(CUfunction function, Pointer[] arguments) {
		// Allocate device output memory
		CUdeviceptr result = getCUdeviceptr((long)size());
		arguments[arguments.length-1] = Pointer.to(result);

		// Set up the kernel parameters: A pointer to an array
		// of pointers which point to the actual values.
		Pointer kernelParameters = Pointer.to(arguments);

		// Call the kernel function.
		int blockSizeX = 256;
		int gridSizeX = (int)Math.ceil((double)size() / blockSizeX);
		cuLaunchKernel(function,
				gridSizeX,  1, 1,      // Grid dimension
				blockSizeX, 1, 1,      // Block dimension
				0, null,               // Shared memory size and stream
				kernelParameters, null // Kernel- and extra parameters
				);
		cuCtxSynchronize();
		return result;
	}

	/**
	 * The extension of the given file name is replaced with "ptx".
	 * If the file with the resulting name does not exist, it is
	 * compiled from the given file using NVCC. The name of the
	 * PTX file is returned.
	 *
	 * @param cuFileName The name of the .CU file
	 * @return The name of the PTX file
	 * @throws IOException If an I/O error occurs
	 */
	private static String preparePtxFile(String cuFileName) throws IOException
	{
		int endIndex = cuFileName.lastIndexOf('.');
		if (endIndex == -1)
		{
			endIndex = cuFileName.length()-1;
		}
		String ptxFileName = cuFileName.substring(0, endIndex+1)+"ptx";
		File ptxFile = new File(ptxFileName);
		if (ptxFile.exists())
		{
			return ptxFileName;
		}

		File cuFile = new File(cuFileName);
		if (!cuFile.exists())
		{
			throw new IOException("Input file not found: "+cuFileName);
		}
		String modelString = "-m"+System.getProperty("sun.arch.data.model");
		String command =
				"nvcc " + modelString + " -ptx "+
						cuFile.getPath()+" -o "+ptxFileName;

		System.out.println("Executing\n"+command);
		Process process = Runtime.getRuntime().exec(command);

		String errorMessage =
				new String(toByteArray(process.getErrorStream()));
		String outputMessage =
				new String(toByteArray(process.getInputStream()));
		int exitValue = 0;
		try
		{
			exitValue = process.waitFor();
		}
		catch (InterruptedException e)
		{
			Thread.currentThread().interrupt();
			throw new IOException(
					"Interrupted while waiting for nvcc output", e);
		}

		if (exitValue != 0)
		{
			System.out.println("nvcc process exitValue "+exitValue);
			System.out.println("errorMessage:\n"+errorMessage);
			System.out.println("outputMessage:\n"+outputMessage);
			throw new IOException(
					"Could not create .ptx file: "+errorMessage);
		}

		System.out.println("Finished creating PTX file");
		return ptxFileName;
	}

	/**
	 * Fully reads the given InputStream and returns it as a byte array
	 *
	 * @param inputStream The input stream to read
	 * @return The byte array containing the data from the input stream
	 * @throws IOException If an I/O error occurs
	 */
	private static byte[] toByteArray(InputStream inputStream)
			throws IOException
			{
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		byte buffer[] = new byte[8192];
		while (true)
		{
			int read = inputStream.read(buffer);
			if (read == -1)
			{
				break;
			}
			baos.write(buffer, 0, read);
		}
		return baos.toByteArray();
			}

}
