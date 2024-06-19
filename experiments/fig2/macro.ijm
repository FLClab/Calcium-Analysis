
// ------------------------------------
// PARAMETERS
// ------------------------------------

HOME=getDirectory("home");
PATH="/Documents/mscts-analysis/notebooks/fig2";
IMAGEID="80-1"
CMAP="coolwarm" // User defined colormap

// ------------------------------------
// /PARAMETERS
// ------------------------------------

run("Close All");

open(HOME + PATH + "/data/" + "movie-" + IMAGEID + ".tif");
rename("movie");
run("Z Project...", "projection=[Max Intensity]");
run("Invert LUTs");

min = 0;
max = 0;
getMinAndMax(min, max);
setMinAndMax(min, (max - min) * 0.25);
rename("max-proj-movie");

wait(50);
run("Flatten");
saveAs("tif", HOME + PATH + "/data/" + "movie-" + IMAGEID + "-xy.tif");
close("movie-" + IMAGEID + "-xy.tif");

open(HOME + PATH + "/data/" + "mask-" + IMAGEID + ".tif");
setMinAndMax(0, 1);
rename("mask");

run("Z-stack Depth Colorcode 0.0.2", "use=[" + CMAP + "] generate=[No] output=[Color (RGB)]");
rename("depth-colorcoded");

run("Z Project...", "projection=[Max Intensity]");
rename("max-proj-mask")

selectImage("max-proj-movie");
run("Add Image...", "image=max-proj-mask x=0 y=0 opacity=100 zero");
rename("xy-overlay");
run("Flatten");
saveAs("tif", HOME + PATH + "/data/" + "movie-" + IMAGEID + "-xy-overlay.tif");

//// ------------------------------------
//// XZ profile
//// ------------------------------------

selectImage("movie");
run("Reslice [/]...", "output=1.000 start=Left");
rename("xz-movie");
run("Z Project...", "projection=[Max Intensity]");
run("Invert LUTs");
setMinAndMax(min, (max - min) * 0.25);

rename("xz-max-proj-movie");

wait(50);
run("Flatten");
saveAs("tif", HOME + PATH + "/data/" + "movie-" + IMAGEID + "-xz.tif");
//close("movie-" + IMAGEID + "-xy.tif");

selectImage("depth-colorcoded");
run("Reslice [/]...", "output=1.000 start=Left");
rename("xz-mask");
run("Z Project...", "projection=[Max Intensity]");
rename("xz-max-proj-mask");

selectImage("xz-max-proj-movie");
run("Add Image...", "image=xz-max-proj-mask x=0 y=0 opacity=100 zero");
rename("xz-overlay");

run("Flatten");
saveAs("tif", HOME + PATH + "/data/" + "movie-" + IMAGEID + "-xz-overlay.tif");

//// ------------------------------------
//// LUT
//// ------------------------------------

newImage("LUTCode", "8-bit ramp", 256, 32, 1);
run("Rotate 90 Degrees Left");
run("coolwarm");
run("Flatten");
saveAs("tif", HOME + PATH + "/data/" + "lut.tif");

