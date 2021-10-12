const contentControl = {
	sliderGroup: document.getElementById("contentSliderGroup"),
	sliders: [],
	sliderPos: [],
	value: [],
	pageSelectors: [],
	currentPage: 0,
	nDim: 512,
	nPage: 26
}

const styleControl = {
	sliderGroup: document.getElementById("styleSliderGroup"),
	sliders: [],
	sliderPos: [],
	value: [],
	pageSelectors: [],
	currentPage: 0,
	nDim: 256,
	nPage: 13
}

const colorControl = {
	controlGroup: document.getElementById("colorControlGroup"),
	colorNum: 11,
	colorNames: [
		"Hair", // 0
		"Light delta", // 1
		"Medium delta", // 2
		"Dark delta", // 3

		"Skin", // 4
		"Shadow", // 5
		"Blush", // 6
		"Mouth", // 7

		"Eyes", // 8
		"Light delta", // 9
		"Dark delta" // 10
	],
	colorPages: [
		{
			name: "Hair",
			ids: [0, 1, 2, 3]
		},
		{
			name: "Face",
			ids: [4, 5, 6, 7]
		},
		{
			name: "Eyes",
			ids: [8, 9, 10]
		}
	],
	colorRecords: [],
	pageSelectors: [],
	currentPage: 0,
	syncEyes: true
}

const displayGroup = document.getElementById("displayGroup");

const artistListElem = document.getElementById("artistList");

var paletteCanvas;
var colorControlCreated = false;
var colorDisplayLoaded = false;
var colorPaletteLoaded = false;

var modelLoaded = false;
var statsLoaded = false;
var allLoaded = false;

var model;
var componentUrlMap;
var stats = {};

var fetchIit = {
	method: 'GET',
	mode: 'cors',
	cache: 'no-cache',
	credentials: 'same-origin',
	headers: {},
	referrer: 'no-referrer'
}

const colorDisplayImage = new Image();
colorDisplayImage.onload = function (event) {
	colorDisplayLoaded = true;
	createColorControl();
}
colorDisplayImage.src = "./color_display.png";

const colorPaletteImage = new Image();
colorPaletteImage.onload = function (event) {
	colorPaletteLoaded = true;
	createColorControl();
}
colorPaletteImage.src = "./color_palette.png";

function createSlider(controlObj, sliderID) {
	let elem = document.createElement("div");
	elem.className = "hbox";
	elem.style.margin = "2px";
	elem.innerHTML = `
		<div class="slider_id dark">0</div>
		<div class="slider_container light">
			<div class="hbox">
				<div class="filler"></div>
				<div class="slider_tickvalue">-2</div>
				<div class="slider_tickvalue">-1</div>
				<div class="slider_tickvalue">0</div>
				<div class="slider_tickvalue">1</div>
				<div class="slider_tickvalue">2</div>
				<div class="filler"></div>
			</div>
			<div class="hbox">
				<div class="filler"></div>
				<div class="slider_tick" style="left:67px;">|</div>
				<div class="slider_tick" style="left:99px;">|</div>
				<div class="slider_tick" style="left:131px;">|</div>
				<div class="slider_tick" style="left:163px;">|</div>
				<div class="slider_tick" style="left:195px;">|</div>
				<div class="filler"></div>
			</div>
			<div class="slider_bar_container">
				<div class="slider_bar medium"></div>
				<div class="slider_marker dark" style="left:84px;"></div>
				<div class="slider_trigger"></div>
			</div>
		</div>
		<div class="slider_value">0.00</div>
	`;
	function func() {setSliderValueByMouse(controlObj, sliderID, event);}
	elem.children[1].children[2].children[2].onmousedown = func;
	elem.children[1].children[2].children[2].onmousemove = func;
	return elem
}

function createSelector(controlObj, pageID, name) {
	let selector = document.createElement("div");
	selector.className = `page_selector light`;
	selector.innerHTML = name;
	selector.onclick = function () {changePage(controlObj, pageID);}
	return selector;
}

function setSliderValueByMouse(controlObj, sliderID, event) {
	if (event.buttons & 1) {
		let t = Math.min(Math.max((event.offsetX - 3) / 160, 0), 1);
		let val = t * 5 - 2.5;
		let loc = Math.min(Math.max(Math.round(event.offsetX) - 3, 0), 160) + 4;
		let dim = controlObj.currentPage * 20 + sliderID;
		controlObj.sliderPos[dim] = loc;
		controlObj.value[dim] = val;
		controlObj.sliders[sliderID].children[2].innerHTML = `${val.toFixed(2)}`;
		controlObj.sliders[sliderID].children[1].children[2].children[1].style.left = `${loc}px`;
	}
}

function changePage(controlObj, pageID) {
	controlObj.pageSelectors[controlObj.currentPage].className = "page_selector light";
	controlObj.pageSelectors[pageID].className = "page_selector dark";
	for (let i = 0; i < 20; i++) {
		let dim = pageID * 20 + i;
		if (dim < controlObj.nDim) {
			controlObj.sliders[i].style.visibility = "visible";
			controlObj.sliders[i].children[0].innerHTML = dim.toString();
			controlObj.sliders[i].children[2].innerHTML = `${controlObj.value[dim].toFixed(2)}`;
			controlObj.sliders[i].children[1].children[2].children[1].style.left = `${controlObj.sliderPos[dim]}px`;
		}
		else {
			controlObj.sliders[i].style.visibility = "hidden";
		}
	}
	controlObj.currentPage = pageID;
}

function createColorPicker(id) {
	let elem = document.createElement("div");
	elem.className = "color_container";
	elem.innerHTML = `
		<div class="color_name dark"></div>
		<div class="color_wheel">
			<canvas width="150" height="150" onmousedown="setChromaticityByMouse(${id}, event);" onmousemove="setChromaticityByMouse(${id}, event);"></canvas>
		</div>
		<div class="color_bar">
			<canvas width="25" height="150" onmousedown="setBrightnessByMouse(${id}, event);" onmousemove="setBrightnessByMouse(${id}, event);"></canvas>
		</div>
	`;
	return elem;
}

function setChromaticityByMouse(id, event) {
	if ((event.buttons & 1) == 0) {
		return;
	}
	elemR = Math.floor(id / 2);
	elemC = id % 2;
	elem = colorControl.controlGroup.children[elemR + 3].children[elemC];

	let x = Math.min(Math.max(Math.round(event.offsetX), 0), 149);
	let y = Math.min(Math.max(Math.round(event.offsetY), 0), 149);
	let paletteContext = paletteCanvas.getContext("2d");
	let imageData = paletteContext.getImageData(x, y, 1, 1);
	let r = imageData.data[0];
	let g = imageData.data[1];
	let b = imageData.data[2];
	let a = imageData.data[3];

	if (a < 255) {
		return;
	}

	colorRecord = colorControl.colorRecords[colorControl.colorPages[colorControl.currentPage].ids[id]];
	colorRecord.currentWheelPos = [x, y];
	colorRecord.currentWheelColor = [r, g, b];
	drawCurrentColor(elem, colorRecord);
}

function setBrightnessByMouse(id, event) {
	if (((event.buttons & 1) == 0) || (event.offsetY < 30)) {
		return;
	}
	elemR = Math.floor(id / 2);
	elemC = id % 2;
	elem = colorControl.controlGroup.children[elemR + 3].children[elemC];

	let y = Math.min(Math.max(Math.round(event.offsetY - 30), 0), 120);

	colorRecord = colorControl.colorRecords[colorControl.colorPages[colorControl.currentPage].ids[id]];
	colorRecord.currentBarPos = y;
	drawCurrentColor(elem, colorRecord);
}

function drawCurrentColor(elem, colorRecord) {
	let wheelContext = elem.children[1].children[0].getContext("2d");
	wheelContext.clearRect(0, 0, 150, 150);
	wheelContext.drawImage(colorDisplayImage, 0, 0)
	x = colorRecord.currentWheelPos[0];
	y = colorRecord.currentWheelPos[1];
	r = colorRecord.currentWheelColor[0];
	g = colorRecord.currentWheelColor[1];
	b = colorRecord.currentWheelColor[2];

	let v = r * 0.299 + g * 0.587 + b * 0.114;
	if (v > 128) {
		wheelContext.strokeStyle = "rgb(0, 0, 0)";
	}
	else {
		wheelContext.strokeStyle = "rgb(255, 255, 255)";
	}
	wheelContext.strokeRect(x - 2, y - 2, 5, 5);

	let barContext = elem.children[2].children[0].getContext("2d");
	barContext.clearRect(0, 0, 25, 150);
	let grad = barContext.createLinearGradient(0, 30, 0, 150);
	grad.addColorStop(0, "rgb(255, 255, 255)");
	grad.addColorStop(0.5, `rgb(${r}, ${g}, ${b})`);
	grad.addColorStop(1, "rgb(0, 0, 0)");
	barContext.fillStyle = grad;
	barContext.fillRect(0, 30, 25, 120);

	imageData = barContext.getImageData(0, 30 + colorRecord.currentBarPos, 1, 1);
	r = imageData.data[0];
	g = imageData.data[1];
	b = imageData.data[2];
	v = r * 0.299 + g * 0.587 + b * 0.114;
	if (v > 128) {
		barContext.strokeStyle = "rgb(0, 0, 0)";
	}
	else {
		barContext.strokeStyle = "rgb(255, 255, 255)";
	}
	barContext.strokeRect(0, 30 + colorRecord.currentBarPos, 25, 1);

	barContext.fillStyle = `rgb(${r}, ${g}, ${b})`;
	barContext.fillRect(0, 0, 25, 25);

	colorRecord.currentColor = [r, g, b];
}

function createColorPageSelector(pageID) {
	let selector = document.createElement("div");
	selector.className = `color_selector light`;
	selector.innerHTML = colorControl.colorPages[pageID].name;
	selector.onclick = function () {changeColorPage(pageID);}
	return selector;
}

function changeColorPage(pageID) {
	colorControl.pageSelectors[colorControl.currentPage].className = "color_selector light";
	colorControl.pageSelectors[pageID].className = "color_selector dark";
	for (let i = 0; i < 4; i++) {
		r = Math.floor(i / 2);
		c = i % 2;
		elem = colorControl.controlGroup.children[r + 3].children[c];
		if (i < colorControl.colorPages[pageID].ids.length) {
			colorID = colorControl.colorPages[pageID].ids[i];
			elem.children[0].innerHTML = colorControl.colorNames[colorID];
			drawCurrentColor(elem, colorControl.colorRecords[colorID]);
			elem.style.visibility = "visible";
		}
		else {
			elem.style.visibility = "hidden";
		}
	}
	colorControl.currentPage = pageID;
}

function colorToColorRecord(r, g, b) {
	let u = (r * 2 - g - b) / Math.sqrt(6);
	let v = (g - b) * Math.SQRT1_2;
	let l = Math.sqrt(u * u + v * v);
	if (l == 0) {
		return {
			currentWheelPos: [75, 75],
			currentWheelColor: [128, 128, 128],
			currentBarPos: Math.round((1 - (r + g + b) / 3) * 120),
			currentColor: [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)]
		}
	}
	let tr = (r - 0.5) * 2;
	let tg = (g - 0.5) * 2;
	let tb = (b - 0.5) * 2;
	if (Math.abs(tr) > Math.abs(tb)) {
		if (Math.abs(tr) > Math.abs(tg)) {
			t1 = tr;
		}
		else {
			t1 = tg;
		}
	}
	else {
		if (Math.abs(tb) > Math.abs(tg)) {
			t1 = tb;
		}
		else {
			t1 = tg;
		}
	}
	let t1p = Math.abs(t1);
	let r1 = 0.5 + (r - 0.5) / t1p;
	let g1 = 0.5 + (g - 0.5) / t1p;
	let b1 = 0.5 + (b - 0.5) / t1p;
	let r0, g0, b0, t0, h;
	if (t1 > 0) {
		t0 = Math.max(1 - r1, 1 - g1, 1 - b1);
		r0 = 1 + (r1 - 1) / t0;
		g0 = 1 + (g1 - 1) / t0;
		b0 = 1 + (b1 - 1) / t0;
		h = (t1p * t0 + 1 - t1p) / 2;
	}
	else {
		t0 = Math.max(r1, g1, b1);
		r0 = r1 / t0;
		g0 = g1 / t0;
		b0 = b1 / t0;
		h = 1 - (t1p * t0 + 1 - t1p) / 2;
	}
	let c = (t1p * t0) / (t1p * t0 + 1 - t1p);
	let px = Math.round(v / l * c * 74.5 + 75);
	let py = Math.round(75 - u / l * c * 74.5);
	let wr = Math.round((0.5 * (1 - c) + r0 * c) * 255);
	let wg = Math.round((0.5 * (1 - c) + g0 * c) * 255);
	let wb = Math.round((0.5 * (1 - c) + b0 * c) * 255);
	let pb = Math.round(h * 120);
	return {
		currentWheelPos: [px, py],
		currentWheelColor: [wr, wg, wb],
		currentBarPos: pb,
		currentColor: [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)]
	}
}

function createColorControl() {
	if (!(colorDisplayLoaded && colorPaletteLoaded)) {
		return
	}
	if (colorControlCreated) {
		return
	}
	paletteCanvas = document.createElement("canvas");
	paletteCanvas.width = 150;
	paletteCanvas.height = 150;
	let paletteContext = paletteCanvas.getContext("2d");
	paletteContext.clearRect(0, 0, 150, 150);
	paletteContext.drawImage(colorPaletteImage, 0, 0);

	colorControl.controlGroup.children[3].append(createColorPicker(0));
	colorControl.controlGroup.children[3].append(createColorPicker(1));
	colorControl.controlGroup.children[4].append(createColorPicker(2));
	colorControl.controlGroup.children[4].append(createColorPicker(3));

	for (let i = 0; i < colorControl.colorNum; i++) {
		colorControl.colorRecords.push({
			currentWheelPos: [75, 75],
			currentWheelColor: [128, 128, 128],
			currentBarPos: 60,
			currentColor: [128, 128, 128]
		})
	}

	for (let i = 0; i < 3; i++) {
		let selector = createColorPageSelector(i);
		colorControl.pageSelectors.push(selector);
		colorControl.controlGroup.children[2].append(selector);
	}

	changeColorPage(0);

	colorControlCreated = true;
}

function checkAllLoaded() {
	allLoaded = modelLoaded && statsLoaded;
	if (!allLoaded) {
		return
	}
	displayGroup.children[0].children[1].disabled = false;
	displayGroup.children[0].children[1].innerHTML = "Generate";
	randomizeContent(0);
	//randomizeStyle(0);
	randomSelectStyle();
	randomizeColor(0);
}

function randomizeContent(limited) {
	if (!allLoaded) {
		return;
	}
	let startDim;
	if (limited) {
		startDim = Math.min(Math.max(contentControl.sliderGroup.children[1].children[2].value, 0), 511) + 1;
		if (startDim >= 511) {
			return;
		}
	}
	else {
		startDim = 0;
	}

	let code = tf.randomNormal([1, 64]).matMul(stats.con_basis).squeeze(0).add(stats.con_mean).clipByValue(-2.5, 2.5);
	let value = code.arraySync();
	for (let i = startDim; i < 512; i++) {
		v = value[i]
		contentControl.value[i] = v;
		contentControl.sliderPos[i] = Math.round((v + 2.5) / 5 * 160) + 4;
	}
	changePage(contentControl, contentControl.currentPage);
}

function getContent() {
	return tf.tensor(contentControl.value).expandDims(0);
}

/*
function randomizeStyle(limited) {
	if (!allLoaded) {
		return;
	}
	let startDim;
	if (limited) {
		startDim = Math.min(Math.max(styleControl.sliderGroup.children[2].children[2].value, 0), 255) + 1;
		if (startDim >= 255) {
			return;
		}
	}
	else {
		startDim = 0;
	}

	let code = tf.randomNormal([1, 64]).matMul(stats.sty_basis).squeeze(0).clipByValue(-2.5, 2.5);
	let value = code.arraySync();
	for (let i = startDim; i < 256; i++) {
		styleControl.value[i] = value[i];
		styleControl.sliderPos[i] = Math.round((value[i] + 2.5) / 5 * 160) + 4;
	}
	changePage(styleControl, styleControl.currentPage);
}
*/

function randomSelectStyle() {
	let artistIndex = Math.floor(Math.random() * stats.artist_names.length);
	let artistStyle = stats.artist_sty[artistIndex]
	for (let i = 0; i < 256; i++) {
		v = Math.min(Math.max(artistStyle[i], -2.5), 2.5)
		styleControl.value[i] = v;
		styleControl.sliderPos[i] = Math.round((v + 2.5) / 5 * 160) + 4;
	}
	changePage(styleControl, styleControl.currentPage);
}

function getStyle() {
	return tf.tensor(styleControl.value).expandDims(0);
}

function clearArtistInput() {
	styleControl.sliderGroup.children[1].children[1].value = ''
}

function setStyleByArtist() {
	let artistName = styleControl.sliderGroup.children[1].children[1].value
	if (artistName == "") {
		artistName = "sayori"
		styleControl.sliderGroup.children[1].children[1].value = artistName
	}
	let artistIndex = stats.artist_names.indexOf(artistName)
	if (artistIndex == -1) {
		styleControl.sliderGroup.children[1].children[2].style.visibility = "visible"
		return
	}
	else {
		styleControl.sliderGroup.children[1].children[2].style.visibility = "hidden"
	}
	let artistStyle = stats.artist_sty[artistIndex]
	for (let i = 0; i < 256; i++) {
		v = Math.min(Math.max(artistStyle[i], -2.5), 2.5)
		styleControl.value[i] = v;
		styleControl.sliderPos[i] = Math.round((v + 2.5) / 5 * 160) + 4;
	}
	changePage(styleControl, styleControl.currentPage);
}

function colorWeightedSum(color, weight) {
	let r = 0;
	let g = 0;
	let b = 0;
	for (let i = 0; i < weight.length; i++) {
		r += color[i][0] * weight[i];
		g += color[i][1] * weight[i];
		b += color[i][2] * weight[i];
	}
	return [r, g, b];
}

function colorMatMul(color, mat) {
	return [
		color[0] * mat[0][0] + color[1] * mat[1][0] + color[2] * mat[2][0] + mat[3][0],
		color[0] * mat[0][1] + color[1] * mat[1][1] + color[2] * mat[2][1] + mat[3][1],
		color[0] * mat[0][2] + color[1] * mat[1][2] + color[2] * mat[2][2] + mat[3][2]
	];
}

function scaledColorDiff(x, y) {
	return [
		(Math.min(Math.max(x[0] - y[0], -1), 1) + 1) / 2,
		(Math.min(Math.max(x[1] - y[1], -1), 1) + 1) / 2,
		(Math.min(Math.max(x[2] - y[2], -1), 1) + 1) / 2
	];
}

function randomizeColor(limited) {
	if (!allLoaded) {
		return;
	}
	let part;
	if (limited) {
		part = colorControl.currentPage;
	}
	else {
		part = -1;
	}

	let color = tf.randomNormal([1, 48]).matMul(stats.clr_basis).squeeze(0).add(stats.clr_mean).reshape([16, 3]).clipByValue(-1, 1);
	let color_value = color.arraySync();
	let t, t1, t2;

	if ((part == -1) || (part == 0)) {
		hair_base = colorWeightedSum([color_value[12], color_value[13], color_value[14], color_value[15]], stats.hair_weight);
		colorControl.colorRecords[0] = colorToColorRecord((hair_base[0] + 1) / 2, (hair_base[1] + 1) / 2, (hair_base[2] + 1) / 2);

		hair_light_base = colorMatMul(hair_base, stats.hair_light_mat);
		t = scaledColorDiff(color_value[14], hair_light_base);
		colorControl.colorRecords[1] = colorToColorRecord(t[0], t[1], t[2]);
		
		hair_medium_base = colorMatMul(hair_base, stats.hair_medium_mat);
		t = scaledColorDiff(color_value[12], hair_medium_base);
		colorControl.colorRecords[2] = colorToColorRecord(t[0], t[1], t[2]);
		
		hair_dark_base = colorMatMul(hair_base, stats.hair_dark_mat);
		t = scaledColorDiff(color_value[13], hair_dark_base);
		colorControl.colorRecords[3] = colorToColorRecord(t[0], t[1], t[2]);
	}

	if ((part == -1) || (part == 1)) {
		colorControl.colorRecords[4] = colorToColorRecord((color_value[11][0] + 1) / 2, (color_value[11][1] + 1) / 2, (color_value[11][2] + 1) / 2);
		colorControl.colorRecords[5] = colorToColorRecord((color_value[10][0] + 1) / 2, (color_value[10][1] + 1) / 2, (color_value[10][2] + 1) / 2);
		colorControl.colorRecords[6] = colorToColorRecord((color_value[9][0] + 1) / 2, (color_value[9][1] + 1) / 2, (color_value[9][2] + 1) / 2);
		colorControl.colorRecords[7] = colorToColorRecord((color_value[8][0] + 1) / 2, (color_value[8][1] + 1) / 2, (color_value[8][2] + 1) / 2);
	}

	if ((part == -1) || (part == 2)) {
		reye_base = colorWeightedSum([color_value[2], color_value[3]], stats.reye_weight);
		leye_base = colorWeightedSum([color_value[6], color_value[7]], stats.leye_weight);
		colorControl.colorRecords[8] = colorToColorRecord((reye_base[0] + leye_base[0] + 2) / 4, (reye_base[1] + leye_base[1] + 2) / 4, (reye_base[2] + leye_base[2] + 2) / 4);

		reye_light_base = colorMatMul(reye_base, stats.reye_light_mat);
		leye_light_base = colorMatMul(leye_base, stats.leye_light_mat);
		t1 = scaledColorDiff(color_value[2], reye_light_base);
		t2 = scaledColorDiff(color_value[6], leye_light_base);
		colorControl.colorRecords[9] = colorToColorRecord((t1[0] + t2[0]) / 2, (t1[1] + t2[1]) / 2, (t1[2] + t2[2]) / 2);
		
		reye_dark_base = colorMatMul(reye_base, stats.reye_dark_mat);
		leye_dark_base = colorMatMul(leye_base, stats.leye_dark_mat);
		t1 = scaledColorDiff(color_value[3], reye_dark_base);
		t2 = scaledColorDiff(color_value[7], leye_dark_base);
		colorControl.colorRecords[10] = colorToColorRecord((t1[0] + t2[0]) / 2, (t1[1] + t2[1]) / 2, (t1[2] + t2[2]) / 2);
	}

	changeColorPage(colorControl.currentPage);
}

function getOneColor(k) {
	return [
		colorControl.colorRecords[k].currentColor[0] / 255,
		colorControl.colorRecords[k].currentColor[1] / 255,
		colorControl.colorRecords[k].currentColor[2] / 255,
	];
}

function getColor() {
	let color = [];

	hair_base = getOneColor(0);
	hair_light_base = colorMatMul(hair_base, stats.hair_light_mat);
	hair_light_diff = getOneColor(1);
	color[14] = [hair_light_base[0] + hair_light_diff[0] - 0.5, hair_light_base[1] + hair_light_diff[1] - 0.5, hair_light_base[2] + hair_light_diff[2] - 0.5];
	hair_medium_base = colorMatMul(hair_base, stats.hair_medium_mat);
	hair_medium_diff = getOneColor(2);
	color[12] = [hair_medium_base[0] + hair_medium_diff[0] - 0.5, hair_medium_base[1] + hair_medium_diff[1] - 0.5, hair_medium_base[2] + hair_medium_diff[2] - 0.5];
	hair_dark_base = colorMatMul(hair_base, stats.hair_dark_mat);
	hair_dark_diff = getOneColor(3);
	color[13] = [hair_dark_base[0] + hair_dark_diff[0] - 0.5, hair_dark_base[1] + hair_dark_diff[1] - 0.5, hair_dark_base[2] + hair_dark_diff[2] - 0.5];
	color[15] = stats.color15;

	color[11] = getOneColor(4);
	color[10] = getOneColor(5);
	color[9] = getOneColor(6);
	color[8] = getOneColor(7);

	eye_base = getOneColor(8);
	reye_light_base = colorMatMul(eye_base, stats.reye_light_mat);
	leye_light_base = colorMatMul(eye_base, stats.leye_light_mat);
	eye_light_diff = getOneColor(9);
	color[2] = [
		(reye_light_base[0] + leye_light_base[0]) / 2 + eye_light_diff[0] - 0.5,
		(reye_light_base[1] + leye_light_base[1]) / 2 + eye_light_diff[1] - 0.5,
		(reye_light_base[2] + leye_light_base[2]) / 2 + eye_light_diff[2] - 0.5
	];
	color[6] = color[2];
	leye_dark_base = colorMatMul(eye_base, stats.leye_dark_mat);
	reye_dark_base = colorMatMul(eye_base, stats.reye_dark_mat);
	eye_dark_diff = getOneColor(10);
	color[3] = [
		(reye_dark_base[0] + leye_dark_base[0]) / 2 + eye_dark_diff[0] - 0.5,
		(reye_dark_base[1] + leye_dark_base[1]) / 2 + eye_dark_diff[1] - 0.5,
		(reye_dark_base[2] + leye_dark_base[2]) / 2 + eye_dark_diff[2] - 0.5
	];
	color[7] = color[3];

	color[0] = stats.color0;
	color[4] = stats.color0;
	color[1] = stats.color1;
	color[5] = stats.color1;

	return tf.tensor(color).expandDims(0);
}

async function generate() {
	console.log("generating...")
	let content = getContent();
	let style = getStyle();
	let color = getColor()
	let noise = tf.zeros([1, 64]);
	let output = await model.executeAsync({content: content, style: style, noise: noise, color: color});
	tf.browser.toPixels(output.squeeze(0).transpose([1, 2, 0]), displayGroup.children[0].children[0]);
	console.log("finished")
}

async function getComponentUrlAsync(name) {
	return componentUrlMap[name];
}

function loadModel() {
	tf.loadGraphModel(componentUrlMap["model.json"], {weightUrlConverter: getComponentUrlAsync, requestInit: fetchIit})
	.then(ret => {
		model = ret;
		modelLoaded = true;
		checkAllLoaded();
		return true;
	})
	.catch(e => {
		console.error("Failed to load model: " + e.message)
	});
}

function loadStats() {
	fetch(componentUrlMap["stats.json"], fetchIit)
	.then(response => {
		if (!response.ok) {
			throw new Error("HTTP error");
		}
		return response.json()
	})
	.then(rawStats => {
		stats = rawStats;
		stats.color0 = [(stats.clr_mean[0] + stats.clr_mean[12] + 2) / 4, (stats.clr_mean[1] + stats.clr_mean[13] + 2) / 4, (stats.clr_mean[2] + stats.clr_mean[14] + 2) / 4];
		stats.color1 = [(stats.clr_mean[2] + stats.clr_mean[15] + 2) / 4, (stats.clr_mean[3] + stats.clr_mean[16] + 2) / 4, (stats.clr_mean[5] + stats.clr_mean[17] + 2) / 4];
		stats.color15 = [(stats.clr_mean[45] + 1) / 2, (stats.clr_mean[46] + 1) / 2, (stats.clr_mean[47] + 1) / 2];
		for (key of ['sty_basis', 'con_mean', 'con_basis', 'clr_mean', 'clr_basis']) {
			stats[key] = tf.tensor(stats[key]);
		}

		artistListString = ""
		for (name of stats.artist_names) {
			artistListString = artistListString + '<option value="' + name + '">'
		}
		artistListElem.innerHTML = artistListString

		statsLoaded = true;
		checkAllLoaded();
		return true;
	});
}

function load() {
	fetch("component_url.json")
	.then(response => {
		if (!response.ok) {
			throw new Error("HTTP error");
		}
		return response.json()
	})
	.then(t => {
		componentUrlMap = t;
		loadModel();
		loadStats();
	})
}

for (controlObj of [contentControl, styleControl]) {
	for (let i = 0; i < 20; i++) {
		slider = createSlider(controlObj, i);
		c = Math.floor(i / 10);
		controlObj.sliderGroup.children[2].children[1 + c].append(slider);
		controlObj.sliders.push(slider);
	}

	for (let i = 0; i < controlObj.nDim; i++) {
		controlObj.value.push(0);
		controlObj.sliderPos.push(84);
	}

	for (let i = 0; i < controlObj.nPage; i++) {
		let selector = createSelector(controlObj, i, (i * 20).toString());
		controlObj.pageSelectors.push(selector);
		controlObj.sliderGroup.children[2].children[0].append(selector);
	}

	changePage(controlObj, 0);
}

load()