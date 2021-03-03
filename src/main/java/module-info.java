
module com.christianfries.teaching.gpu {
	exports com.christianfries.teaching.gpu;
	exports com.christianfries.opencl.examples;
	
	requires javafx.controls;
	requires javafx.base;
	requires transitive javafx.graphics;
	requires javafx.swing;

	requires java.logging;
	requires java.management;
	requires java.sql;
	requires jcuda;
	requires jocl;
	requires jcublas;
	requires jcurand;
}