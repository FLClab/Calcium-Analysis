run("Close All");

dir = "/Users/Anthony/Documents/mscts-analysis/notebooks/fig6/experiments/panels/individual/examples/"

files = newArray();
files = listFiles(files, dir); 

for (i=0; i < files.length; i ++) {
//for (i=0; i < 5; i ++) {	
	filename=files[i];
	open(filename);
	title=getTitle();
	
	run("   Grays ");
	if (indexOf(title.toLowerCase(), "max-proj-pred") <= 0) {
		run("Invert LUT");
		min = 0;
		max = 2;
	} else {
		min=0;
		max=1;
	}
	
	setMinAndMax(min, max);		
	
	if (indexOf(title.toLowerCase(), "max-proj.tif") >= 0){
		event_only=filename.replace("max-proj.tif", "max-proj-pred_event-only.tif");
		open(event_only);
		setAutoThreshold("Default dark");
		run("Create Selection");
		roiManager("Add");
		selectImage(title);
		roiManager("Set Color", "#00aa00");
		roiManager("Set Line Width", 1);
		run("From ROI Manager");
	}
	
	wait(100);
	
	selectImage(title);
	run("Flatten");
	newtitle=title.replace(".tif", "_rgb.tif");
	rename(newtitle);
	
	newfilename = filename.replace(".tif", "_rgb.tif");
	saveAs("Tiff", newfilename);
	
	run("Close All");
	if (roiManager("count") > 0){
		roiManager("deselect");
		roiManager("Delete");
	}
}

function listFiles(files, dir) {
 list = getFileList(dir);
 for (i=0; i<list.length; i++) {
    if (endsWith(list[i], "/")){
       files = listFiles(files, ""+dir+list[i]);
    } else {
       filename = dir + list[i];
	   if (endsWith(filename, ".tif") & !endsWith(filename, "_rgb.tif") & !endsWith(filename, "_event-only.tif")){
	      filepath = newArray(filename);
	      files = Array.concat(files, filepath);
	   }
    }
 }
 return files;
}
