/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christian-fries.de.
 *
 * Created on 09.02.2006
 */


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
import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;

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
		cuModuleGetFunction(add, module, "add");
	}


	// Need to ref to data here
	private long size;
	
	/**
	 * Create a stochastic random variable.
	 *
	 * @param realisations the vector of realizations.
	 */
	public RandomVariableSimpleCuda(float[] realisations) {
		super();
		this.size = realisations.length;
		// DO STUFF HERE
	}


	@Override
	protected void finalize() throws Throwable {
		// CLEAN UP STUFF HERE
	}


	@Override
	public long size() {
		return size;
	}

	@Override
	public float[] getRealizations() {
		// Return data here
		return null;
	}

	@Override
	public RandomVariableSimpleInterface add(RandomVariableSimpleInterface randomVariable) {
		// Do stuff here
		return null;
	}

	@Override
	public RandomVariableSimpleInterface div(
			RandomVariableSimpleInterface randomVariable) {
		// TODO Auto-generated method stub
		return null;
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
