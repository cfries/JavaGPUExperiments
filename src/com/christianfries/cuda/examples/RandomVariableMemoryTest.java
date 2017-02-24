/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christian-fries.de.
 *
 * Created on 24.02.2017
 */

package com.christianfries.cuda.examples;

import org.junit.Assert;
import org.junit.Test;

/**
 * Test cases for the class net.finmath.montecarlo.RandomVariable.
 * 
 * @author Christian Fries
 * @see net.finmath.montecarlo.RandomVariable
 */
public class RandomVariableMemoryTest {

	@Test
	public void testRandomVariableStochastic() throws InterruptedException {
		int numberOfPath = 1000000;
		int numberOfRepetitions = 10;
		
		float[] values = new float[numberOfPath];
		for(int i=0; i<values.length; i++) values[i] = i;
		
		RandomVariableSimpleInterface randomVariable = new RandomVariableSimpleCuda(values);

		// Perform some calculations
		RandomVariableSimpleInterface result = null;
		for(int j=0; j<numberOfRepetitions; j++) {
			result = randomVariable.add(randomVariable);
		}
		
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
		
		// The random variable has average value numberOfPath-1
		Assert.assertEquals(numberOfPath-1.0  /* expected */, average /* actual */, 1E-6 /* tolerance */);
	}
}
