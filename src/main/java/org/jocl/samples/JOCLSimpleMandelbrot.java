/*
 * JOCL - Java bindings for OpenCL
 *
 * Copyright 2009 Marco Hutter - http://www.jocl.org/
 */

package org.jocl.samples;

import static org.jocl.CL.CL_CONTEXT_PLATFORM;
import static org.jocl.CL.CL_DEVICE_TYPE_ALL;
import static org.jocl.CL.CL_MEM_READ_WRITE;
import static org.jocl.CL.CL_MEM_WRITE_ONLY;
import static org.jocl.CL.CL_TRUE;
import static org.jocl.CL.clBuildProgram;
import static org.jocl.CL.clCreateBuffer;
import static org.jocl.CL.clCreateCommandQueue;
import static org.jocl.CL.clCreateContext;
import static org.jocl.CL.clCreateKernel;
import static org.jocl.CL.clCreateProgramWithSource;
import static org.jocl.CL.clEnqueueNDRangeKernel;
import static org.jocl.CL.clEnqueueReadBuffer;
import static org.jocl.CL.clEnqueueWriteBuffer;
import static org.jocl.CL.clGetDeviceIDs;
import static org.jocl.CL.clGetPlatformIDs;
import static org.jocl.CL.clSetKernelArg;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Point;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;

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
 * A class that uses a simple OpenCL kernel to compute the
 * Mandelbrot set and displays it in an image
 */
public class JOCLSimpleMandelbrot
{
	/**
	 * Entry point for this sample.
	 *
	 * @param args not used
	 */
	public static void main(final String args[])
	{
		SwingUtilities.invokeLater(new Runnable()
		{
			public void run()
			{
				new JOCLSimpleMandelbrot(1000,1000);
			}
		});
	}

	/**
	 * The image which will contain the Mandelbrot pixel data
	 */
	private final BufferedImage image;

	/**
	 * The width of the image
	 */
	private int sizeX = 0;

	/**
	 * The height of the image
	 */
	private int sizeY = 0;

	/**
	 * The component which is used for rendering the image
	 */
	private final JComponent imageComponent;

	/**
	 * The OpenCL context
	 */
	private cl_context context;

	/**
	 * The OpenCL command queue
	 */
	private cl_command_queue commandQueue;

	/**
	 * The OpenCL kernel which will actually compute the Mandelbrot
	 * set and store the pixel data in a CL memory object
	 */
	private cl_kernel kernel;

	/**
	 * The OpenCL memory object which stores the pixel data
	 */
	private cl_mem pixelMem;

	/**
	 * An OpenCL memory object which stores a nifty color map,
	 * encoded as integers combining the RGB components of
	 * the colors.
	 */
	private cl_mem colorMapMem;

	/**
	 * The color map which will be copied to OpenCL for filling
	 * the PBO.
	 */
	private int colorMap[];

	/**
	 * The minimum x-value of the area in which the Mandelbrot
	 * set should be computed
	 */
	private float x0 = -2f;

	/**
	 * The minimum y-value of the area in which the Mandelbrot
	 * set should be computed
	 */
	private float y0 = -1.3f;

	/**
	 * The maximum x-value of the area in which the Mandelbrot
	 * set should be computed
	 */
	private float x1 = 0.6f;

	/**
	 * The maximum y-value of the area in which the Mandelbrot
	 * set should be computed
	 */
	private float y1 = 1.3f;


	/**
	 * Creates the JOCLSimpleMandelbrot sample with the given
	 * width and height
	 */
	public JOCLSimpleMandelbrot(final int width, final int height)
	{
		this.sizeX = width;
		this.sizeY = height;

		// Create the image and the component that will paint the image
		image = new BufferedImage(sizeX, sizeY, BufferedImage.TYPE_INT_RGB);
		imageComponent = new JPanel()
		{
			private static final long serialVersionUID = 1L;
			public void paintComponent(final Graphics g)
			{
				super.paintComponent(g);
				g.drawImage(image, 0,0,this);
			}
		};

		// Initialize the mouse interaction
		initInteraction();

		// Initialize OpenCL
		initCL();

		// Initial image update
		updateImage();

		// Create the main frame
		final JFrame frame = new JFrame("JOCL Simple Mandelbrot");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setLayout(new BorderLayout());
		imageComponent.setPreferredSize(new Dimension(width, height));
		frame.add(imageComponent, BorderLayout.CENTER);
		frame.pack();

		frame.setVisible(true);
	}

	/**
	 * Initialize OpenCL: Create the context, the command queue
	 * and the kernel.
	 */
	private void initCL()
	{
		final int platformIndex = 0;
		final long deviceType = CL_DEVICE_TYPE_ALL;
		final int deviceIndex = 2;

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
		context = clCreateContext(
				contextProperties, 1, new cl_device_id[]{device},
				null, null, null);

		// Create a command-queue for the selected device
		commandQueue =
				clCreateCommandQueue(context, device, 0, null);

		// Program Setup
		final String source = readFile("/SimpleMandelbrot.cl");

		// Create the program
		final cl_program cpProgram = clCreateProgramWithSource(context, 1,
				new String[]{ source }, null, null);

		// Build the program
		clBuildProgram(cpProgram, 0, null, "-cl-mad-enable", null, null);

		// Create the kernel
		kernel = clCreateKernel(cpProgram, "computeMandelbrot", null);

		// Create the memory object which will be filled with the
		// pixel data
		pixelMem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				sizeX * sizeY * Sizeof.cl_uint, null, null);

		// Create and fill the memory object containing the color map
		initColorMap(32, Color.RED, Color.GREEN, Color.BLUE);
		colorMapMem = clCreateBuffer(context, CL_MEM_READ_WRITE,
				colorMap.length * Sizeof.cl_uint, null, null);
		clEnqueueWriteBuffer(commandQueue, colorMapMem, true, 0,
				colorMap.length * Sizeof.cl_uint, Pointer.to(colorMap), 0, null, null);
	}

	/**
	 * Helper function which reads the file with the given name and returns
	 * the contents of this file as a String. Will exit the application
	 * if the file can not be read.
	 *
	 * @param fileName The name of the file to read.
	 * @return The contents of the file
	 */
	private String readFile(final String fileName)
	{
		try(BufferedReader br = new BufferedReader(new InputStreamReader(getClass().getResourceAsStream(fileName))))
		{
			final StringBuffer sb = new StringBuffer();
			String line = null;
			while (true)
			{
				line = br.readLine();
				if (line == null)
				{
					break;
				}
				sb.append(line).append("\n");
			}
			return sb.toString();
		}
		catch (final IOException e)
		{
			e.printStackTrace();
			System.exit(1);
			return null;
		}
	}

	/**
	 * Creates the colorMap array which contains RGB colors as integers,
	 * interpolated through the given colors with colors.length * stepSize
	 * steps
	 *
	 * @param stepSize The number of interpolation steps between two colors
	 * @param colors The colors for the map
	 */
	private void initColorMap(final int stepSize, final Color ... colors)
	{
		colorMap = new int[stepSize*colors.length];
		int index = 0;
		for (int i=0; i<colors.length-1; i++)
		{
			final Color c0 = colors[i];
			final int r0 = c0.getRed();
			final int g0 = c0.getGreen();
			final int b0 = c0.getBlue();

			final Color c1 = colors[i+1];
			final int r1 = c1.getRed();
			final int g1 = c1.getGreen();
			final int b1 = c1.getBlue();

			final int dr = r1-r0;
			final int dg = g1-g0;
			final int db = b1-b0;

			for (int j=0; j<stepSize; j++)
			{
				final float alpha = (float)j / (stepSize-1);
				final int r = (int)(r0 + alpha * dr);
				final int g = (int)(g0 + alpha * dg);
				final int b = (int)(b0 + alpha * db);
				final int rgb =
						(r << 16) |
						(g <<  8) |
						(b <<  0);
				colorMap[index++] = rgb;
			}
		}
	}


	/**
	 * Attach the mouse- and mouse wheel listeners to the glComponent
	 * which allow zooming and panning the fractal
	 */
	private void initInteraction()
	{
		final Point previousPoint = new Point();

		imageComponent.addMouseMotionListener(new MouseMotionListener()
		{
			@Override
			public void mouseDragged(final MouseEvent e)
			{
				final int dx = previousPoint.x - e.getX();
				final int dy = previousPoint.y - e.getY();

				final float wdx = x1-x0;
				final float wdy = y1-y0;

				x0 += (dx / 150.0f) * wdx;
				x1 += (dx / 150.0f) * wdx;

				y0 += (dy / 150.0f) * wdy;
				y1 += (dy / 150.0f) * wdy;

				previousPoint.setLocation(e.getX(), e.getY());

				updateImage();
			}

			@Override
			public void mouseMoved(final MouseEvent e)
			{
				previousPoint.setLocation(e.getX(), e.getY());
			}

		});

		imageComponent.addMouseWheelListener(new MouseWheelListener()
		{
			@Override
			public void mouseWheelMoved(final MouseWheelEvent e)
			{
				final float dx = x1-x0;
				final float dy = y1-y0;
				final float delta = e.getWheelRotation() / 20.0f;
				x0 += delta * dx;
				x1 -= delta * dx;
				y0 += delta * dy;
				y1 -= delta * dy;

				updateImage();
			}
		});
	}


	/**
	 * Execute the kernel function and read the resulting pixel data
	 * into the BufferedImage
	 */
	private void updateImage()
	{
		// Set work size and execute the kernel
		final long globalWorkSize[] = new long[2];
		globalWorkSize[0] = sizeX;
		globalWorkSize[1] = sizeY;

		final int maxIterations = 2000;
		clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(pixelMem));
		clSetKernelArg(kernel, 1, Sizeof.cl_uint, Pointer.to(new int[]{ sizeX }));
		clSetKernelArg(kernel, 2, Sizeof.cl_uint, Pointer.to(new int[]{ sizeY }));
		clSetKernelArg(kernel, 3, Sizeof.cl_float, Pointer.to(new float[]{ x0 }));
		clSetKernelArg(kernel, 4, Sizeof.cl_float, Pointer.to(new float[]{ y0 }));
		clSetKernelArg(kernel, 5, Sizeof.cl_float, Pointer.to(new float[]{ x1 }));
		clSetKernelArg(kernel, 6, Sizeof.cl_float, Pointer.to(new float[]{ y1 }));
		clSetKernelArg(kernel, 7, Sizeof.cl_int, Pointer.to(new int[]{ maxIterations }));
		clSetKernelArg(kernel, 8, Sizeof.cl_mem, Pointer.to(colorMapMem));
		clSetKernelArg(kernel, 9, Sizeof.cl_int, Pointer.to(new int[]{ colorMap.length }));

		clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, globalWorkSize, null, 0, null, null);

		// Read the pixel data into the BufferedImage
		final DataBufferInt dataBuffer = (DataBufferInt)image.getRaster().getDataBuffer();
		final int data[] = dataBuffer.getData();
		clEnqueueReadBuffer(commandQueue, pixelMem, CL_TRUE, 0, Sizeof.cl_int * sizeY * sizeX, Pointer.to(data), 0, null, null);

		imageComponent.repaint();
	}
}
