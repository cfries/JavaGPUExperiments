/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christian-fries.de.
 *
 * Created on 10.02.2004
 */


import org.junit.Assert;
import org.junit.Test;

/**
 * Test cases for the class net.finmath.montecarlo.RandomVariable.
 * 
 * @author Christian Fries
 * @see net.finmath.montecarlo.RandomVariable
 */
public class RandomVariableTests {

	@Test
	public void testRandomVariableStochastic() throws InterruptedException {
		RandomVariableSimpleInterface randomVariable1 = new RandomVariableSimpleCuda(new float[] {-4.0f, -2.0f, 0.0f, 2.0f, 4.0f} );
		RandomVariableSimpleInterface randomVariable2 = new RandomVariableSimpleCuda(new float[] { 4.0f,  4.0f, 4.0f, 4.0f, 4.0f} );
		RandomVariableSimpleInterface randomVariable3 = new RandomVariableSimpleCuda(new float[] { 2.0f,  2.0f, 2.0f, 2.0f, 2.0f} );

		// Perform some calculations
		RandomVariableSimpleInterface result = randomVariable1.add(randomVariable2).div(randomVariable3);

		float[] resultArray = result.getRealizations();
		
		double sum = 0.0;
		double sumOfSquares = 0.0;
		for(int i=0; i<resultArray.length; i++) {
			float value = resultArray[i];
			sum += value;
			sumOfSquares += value*value;
		}
		double average = sum / resultArray.length;
		double variance = sumOfSquares / resultArray.length - average*average;
		
		// The random variable has average value 2.0
		Assert.assertEquals(2.0  /* expected */, average /* actual */, 1E-6 /* tolerance */);

		// The random variable has variance value 2.0 = (4 + 1 + 0 + 1 + 4) / 5
		Assert.assertEquals(2.0  /* expected */, variance /* actual */, 1E-6 /* tolerance */);
	}
}
