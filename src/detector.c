#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/core/core_c.h"
//#include "opencv2/core/core.hpp"
#include "opencv2/core/version.hpp"
#include "opencv2/imgproc/imgproc_c.h"

#ifndef CV_VERSION_EPOCH
#include "opencv2/videoio/videoio_c.h"
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR)""CVAUX_STR(CV_VERSION_MINOR)""CVAUX_STR(CV_VERSION_REVISION)
#pragma comment(lib, "opencv_world" OPENCV_VERSION ".lib")
#else
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_EPOCH)""CVAUX_STR(CV_VERSION_MAJOR)""CVAUX_STR(CV_VERSION_MINOR)
#pragma comment(lib, "opencv_core" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_highgui" OPENCV_VERSION ".lib")
#endif

IplImage* draw_train_chart(float max_img_loss, int max_batches, int number_of_lines, int img_size);
void draw_train_loss(IplImage* img, int img_size, float avg_loss, float max_img_loss, int current_batch, int max_batches);
#endif	// OPENCV

static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};

void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, int dont_show)
{
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.list");
    char *backup_directory = option_find_str(options, "backup", "/backup/");

    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network *nets = calloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = parse_network_cfg(cfgfile);
        if(weightfile){
            load_weights(&nets[i], weightfile);
        }
        if(clear) *nets[i].seen = 0;
        nets[i].learning_rate *= ngpus;
    }
    srand(time(0));
    network net = nets[0];

    int imgs = net.batch * net.subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    data train, buffer;

    layer l = net.layers[net.n - 1];

    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

	int init_w = net.w;
	int init_h = net.h;
	int iter_save;
	iter_save = get_current_batch(net);

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
	args.small_object = net.small_object;
    args.d = &buffer;
    args.type = DETECTION_DATA;
	args.threads = 64;	// 8

    args.angle = net.angle;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;

#ifdef OPENCV
	IplImage* img = NULL;
	float max_img_loss = 5;
	int number_of_lines = 100;
	int img_size = 1000;
	if (!dont_show)
		img = draw_train_chart(max_img_loss, net.max_batches, number_of_lines, img_size);
#endif	//OPENCV

    pthread_t load_thread = load_data(args);
    clock_t time;
    int count = 0;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net.max_batches){
		if(l.random && count++%10 == 0){
            printf("Resizing\n");
			int dim = (rand() % 12 + (init_w/32 - 5)) * 32;	// +-160
            //if (get_current_batch(net)+100 > net.max_batches) dim = 544;
            //int dim = (rand() % 4 + 16) * 32;
            printf("%d\n", dim);
            args.w = dim;
            args.h = dim;

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            for(i = 0; i < ngpus; ++i){
                resize_network(nets + i, dim, dim);
            }
            net = nets[0];
        }
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        /*
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[10] + 1 + k*5);
           if(!b.x) break;
           printf("loaded: %f %f %f %f\n", b.x, b.y, b.w, b.h);
           }
           image im = float_to_image(448, 448, 3, train.X.vals[10]);
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[10] + 1 + k*5);
           printf("%d %d %d %d\n", truth.x, truth.y, truth.w, truth.h);
           draw_bbox(im, b, 8, 1,0,0);
           }
           save_image(im, "truth11");
         */
		if (i % 10 == 0)
        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        i = get_current_batch(net);
		if (i%10==0)
        printf("\n %d: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);

#ifdef OPENCV
		if(!dont_show)
			draw_train_loss(img, img_size, avg_loss, max_img_loss, i, net.max_batches);
#endif	// OPENCV

		//if (i % 1000 == 0 || (i < 1000 && i % 100 == 0)) {
		//if (i % 100 == 0) {
		if(i >= (iter_save + 500)) {
			iter_save = i;
#ifdef GPU
			if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
			char buff[256];
			sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
			save_weights(net, buff);
		}
        free_data(train);
    }
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);

	//cvReleaseImage(&img);
	//cvDestroyAllWindows();
}


static int get_coco_image_id(char *filename)
{
	char *p = strrchr(filename, '/');
	char *c = strrchr(filename, '_');
	if (c) p = c;
	return atoi(p + 1);
}

static void print_cocos(FILE *fp, char *image_path, detection *dets, int num_boxes, int classes, int w, int h)
{
	int i, j;
	int image_id = get_coco_image_id(image_path);
	for (i = 0; i < num_boxes; ++i) {
		float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
		float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
		float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
		float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

		if (xmin < 0) xmin = 0;
		if (ymin < 0) ymin = 0;
		if (xmax > w) xmax = w;
		if (ymax > h) ymax = h;

		float bx = xmin;
		float by = ymin;
		float bw = xmax - xmin;
		float bh = ymax - ymin;

		for (j = 0; j < classes; ++j) {
			if (dets[i].prob[j]) fprintf(fp, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id, coco_ids[j], bx, by, bw, bh, dets[i].prob[j]);
		}
	}
}

void print_detector_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h)
{
	int i, j;
	for (i = 0; i < total; ++i) {
		float xmin = dets[i].bbox.x - dets[i].bbox.w / 2. + 1;
		float xmax = dets[i].bbox.x + dets[i].bbox.w / 2. + 1;
		float ymin = dets[i].bbox.y - dets[i].bbox.h / 2. + 1;
		float ymax = dets[i].bbox.y + dets[i].bbox.h / 2. + 1;

		if (xmin < 1) xmin = 1;
		if (ymin < 1) ymin = 1;
		if (xmax > w) xmax = w;
		if (ymax > h) ymax = h;

		for (j = 0; j < classes; ++j) {
			if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
				xmin, ymin, xmax, ymax);
		}
	}
}

void print_imagenet_detections(FILE *fp, int id, detection *dets, int total, int classes, int w, int h)
{
	int i, j;
	for (i = 0; i < total; ++i) {
		float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
		float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
		float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
		float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

		if (xmin < 0) xmin = 0;
		if (ymin < 0) ymin = 0;
		if (xmax > w) xmax = w;
		if (ymax > h) ymax = h;

		for (j = 0; j < classes; ++j) {
			int class = j;
			if (dets[i].prob[class]) fprintf(fp, "%d %d %f %f %f %f %f\n", id, j + 1, dets[i].prob[class],
				xmin, ymin, xmax, ymax);
		}
	}
}

void validate_detector(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
	int j;
	list *options = read_data_cfg(datacfg);
	char *valid_images = option_find_str(options, "valid", "data/train.list");
	char *name_list = option_find_str(options, "names", "data/names.list");
	char *prefix = option_find_str(options, "results", "results");
	char **names = get_labels(name_list);
	char *mapf = option_find_str(options, "map", 0);
	int *map = 0;
	if (mapf) map = read_map(mapf);

	network net = parse_network_cfg_custom(cfgfile, 1);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);
	fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
	srand(time(0));

	list *plist = get_paths(valid_images);
	char **paths = (char **)list_to_array(plist);

	layer l = net.layers[net.n - 1];
	int classes = l.classes;

	char buff[1024];
	char *type = option_find_str(options, "eval", "voc");
	FILE *fp = 0;
	FILE **fps = 0;
	int coco = 0;
	int imagenet = 0;
	if (0 == strcmp(type, "coco")) {
		if (!outfile) outfile = "coco_results";
		snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
		fp = fopen(buff, "w");
		fprintf(fp, "[\n");
		coco = 1;
	}
	else if (0 == strcmp(type, "imagenet")) {
		if (!outfile) outfile = "imagenet-detection";
		snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
		fp = fopen(buff, "w");
		imagenet = 1;
		classes = 200;
	}
	else {
		if (!outfile) outfile = "comp4_det_test_";
		fps = calloc(classes, sizeof(FILE *));
		for (j = 0; j < classes; ++j) {
			snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
			fps[j] = fopen(buff, "w");
		}
	}


	int m = plist->size;
	int i = 0;
	int t;

	float thresh = .005;
	float nms = .45;

	int nthreads = 4;
	image *val = calloc(nthreads, sizeof(image));
	image *val_resized = calloc(nthreads, sizeof(image));
	image *buf = calloc(nthreads, sizeof(image));
	image *buf_resized = calloc(nthreads, sizeof(image));
	pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

	load_args args = { 0 };
	args.w = net.w;
	args.h = net.h;
	args.type = IMAGE_DATA;
	//args.type = LETTERBOX_DATA;

	for (t = 0; t < nthreads; ++t) {
		args.path = paths[i + t];
		args.im = &buf[t];
		args.resized = &buf_resized[t];
		thr[t] = load_data_in_thread(args);
	}
	time_t start = time(0);
	for (i = nthreads; i < m + nthreads; i += nthreads) {
		fprintf(stderr, "%d\n", i);
		for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
			pthread_join(thr[t], 0);
			val[t] = buf[t];
			val_resized[t] = buf_resized[t];
		}
		for (t = 0; t < nthreads && i + t < m; ++t) {
			args.path = paths[i + t];
			args.im = &buf[t];
			args.resized = &buf_resized[t];
			thr[t] = load_data_in_thread(args);
		}
		for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
			char *path = paths[i + t - nthreads];
			char *id = basecfg(path);
			float *X = val_resized[t].data;
			network_predict(net, X);
			int w = val[t].w;
			int h = val[t].h;
			int nboxes = 0;
			int letterbox = (args.type == LETTERBOX_DATA);
			detection *dets = get_network_boxes(&net, w, h, thresh, .5, map, 0, &nboxes, letterbox);
			if (nms) do_nms_sort_v3(dets, nboxes, classes, nms);
			if (coco) {
				print_cocos(fp, path, dets, nboxes, classes, w, h);
			}
			else if (imagenet) {
				print_imagenet_detections(fp, i + t - nthreads + 1, dets, nboxes, classes, w, h);
			}
			else {
				print_detector_detections(fps, id, dets, nboxes, classes, w, h);
			}
			free_detections(dets, nboxes);
			free(id);
			free_image(val[t]);
			free_image(val_resized[t]);
		}
	}
	for (j = 0; j < classes; ++j) {
		if (fps) fclose(fps[j]);
	}
	if (coco) {
		fseek(fp, -2, SEEK_CUR);
		fprintf(fp, "\n]\n");
		fclose(fp);
	}
	fprintf(stderr, "Total Detection Time: %f Seconds\n", time(0) - start);
}

void validate_detector_recall(char *datacfg, char *cfgfile, char *weightfile)
{
	network net = parse_network_cfg_custom(cfgfile, 1);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);
	srand(time(0));

	//list *plist = get_paths("data/coco_val_5k.list");
	list *options = read_data_cfg(datacfg);
	char *valid_images = option_find_str(options, "valid", "data/train.txt");
	list *plist = get_paths(valid_images);
	char **paths = (char **)list_to_array(plist);

	layer l = net.layers[net.n - 1];

	int j, k;

	int m = plist->size;
	int i = 0;

	float thresh = .001;
	float iou_thresh = .5;
	float nms = .4;

	int total = 0;
	int correct = 0;
	int proposals = 0;
	float avg_iou = 0;

	for (i = 0; i < m; ++i) {
		char *path = paths[i];
		image orig = load_image_color(path, 0, 0);
		image sized = resize_image(orig, net.w, net.h);
		char *id = basecfg(path);
		network_predict(net, sized.data);
		int nboxes = 0;
		int letterbox = 0;
		detection *dets = get_network_boxes(&net, sized.w, sized.h, thresh, .5, 0, 1, &nboxes, letterbox);
		if (nms) do_nms_obj_v3(dets, nboxes, 1, nms);

		char labelpath[4096];
		find_replace(path, "images", "labels", labelpath);
		find_replace(labelpath, "JPEGImages", "labels", labelpath);
		find_replace(labelpath, ".jpg", ".txt", labelpath);
		find_replace(labelpath, ".JPEG", ".txt", labelpath);

		int num_labels = 0;
		box_label *truth = read_boxes(labelpath, &num_labels);
		for (k = 0; k < nboxes; ++k) {
			if (dets[k].objectness > thresh) {
				++proposals;
			}
		}
		for (j = 0; j < num_labels; ++j) {
			++total;
			box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
			float best_iou = 0;
			for (k = 0; k < l.w*l.h*l.n; ++k) {
				float iou = box_iou(dets[k].bbox, t);
				if (dets[k].objectness > thresh && iou > best_iou) {
					best_iou = iou;
				}
			}
			avg_iou += best_iou;
			if (best_iou > iou_thresh) {
				++correct;
			}
		}

		fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals / (i + 1), avg_iou * 100 / total, 100.*correct / total);
		free(id);
		free_image(orig);
		free_image(sized);
	}
}

typedef struct {
	box b;
	float p;
	int class_id;
	int image_index;
	int truth_flag;
	int unique_truth_index;
} box_prob;

int detections_comparator(const void *pa, const void *pb)
{
	box_prob a = *(box_prob *)pa;
	box_prob b = *(box_prob *)pb;
	float diff = a.p - b.p;
	if (diff < 0) return 1;
	else if (diff > 0) return -1;
	return 0;
}

void validate_detector_map(char *datacfg, char *cfgfile, char *weightfile, float thresh_calc_avg_iou)
{
	int j;
	list *options = read_data_cfg(datacfg);
	char *valid_images = option_find_str(options, "valid", "data/train.txt");
	char *difficult_valid_images = option_find_str(options, "difficult", NULL);
	char *name_list = option_find_str(options, "names", "data/names.list");
	char **names = get_labels(name_list);
	char *mapf = option_find_str(options, "map", 0);
	int *map = 0;
	if (mapf) map = read_map(mapf);

	network net = parse_network_cfg_custom(cfgfile, 1);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);
	srand(time(0));

	list *plist = get_paths(valid_images);
	char **paths = (char **)list_to_array(plist);

	char **paths_dif = NULL;
	if (difficult_valid_images) {
		list *plist_dif = get_paths(difficult_valid_images);
		paths_dif = (char **)list_to_array(plist_dif);
	}
	

	layer l = net.layers[net.n - 1];
	int classes = l.classes;

	int m = plist->size;
	int i = 0;
	int t;

	const float thresh = .005;
	const float nms = .45;
	const float iou_thresh = 0.5;

	int nthreads = 4;
	image *val = calloc(nthreads, sizeof(image));
	image *val_resized = calloc(nthreads, sizeof(image));
	image *buf = calloc(nthreads, sizeof(image));
	image *buf_resized = calloc(nthreads, sizeof(image));
	pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

	load_args args = { 0 };
	args.w = net.w;
	args.h = net.h;
	args.type = IMAGE_DATA;
	//args.type = LETTERBOX_DATA;

	//const float thresh_calc_avg_iou = 0.24;
	float avg_iou = 0;
	int tp_for_thresh = 0;
	int fp_for_thresh = 0;

	box_prob *detections = calloc(1, sizeof(box_prob));
	int detections_count = 0;
	int unique_truth_count = 0;

	int *truth_classes_count = calloc(classes, sizeof(int));

	for (t = 0; t < nthreads; ++t) {
		args.path = paths[i + t];
		args.im = &buf[t];
		args.resized = &buf_resized[t];
		thr[t] = load_data_in_thread(args);
	}
	time_t start = time(0);
	for (i = nthreads; i < m + nthreads; i += nthreads) {
		fprintf(stderr, "%d\n", i);
		for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
			pthread_join(thr[t], 0);
			val[t] = buf[t];
			val_resized[t] = buf_resized[t];
		}
		for (t = 0; t < nthreads && i + t < m; ++t) {
			args.path = paths[i + t];
			args.im = &buf[t];
			args.resized = &buf_resized[t];
			thr[t] = load_data_in_thread(args);
		}
		for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
			const int image_index = i + t - nthreads;
			char *path = paths[image_index];
			char *id = basecfg(path);
			float *X = val_resized[t].data;
			network_predict(net, X);

			int nboxes = 0;
			int letterbox = (args.type == LETTERBOX_DATA);
			float hier_thresh = 0;
			detection *dets = get_network_boxes(&net, 1, 1, thresh, hier_thresh, 0, 1, &nboxes, letterbox);
			if (nms) do_nms_sort_v3(dets, nboxes, l.classes, nms);

			char labelpath[4096];
			find_replace(path, "images", "labels", labelpath);
			find_replace(labelpath, "JPEGImages", "labels", labelpath);
			find_replace(labelpath, ".jpg", ".txt", labelpath);
			find_replace(labelpath, ".JPEG", ".txt", labelpath);
			find_replace(labelpath, ".png", ".txt", labelpath);
			int num_labels = 0;
			box_label *truth = read_boxes(labelpath, &num_labels);
			int i, j;
			for (j = 0; j < num_labels; ++j) {
				truth_classes_count[truth[j].id]++;
			}

			// difficult
			box_label *truth_dif = NULL;
			int num_labels_dif = 0;
			if (paths_dif)
			{
				char *path_dif = paths_dif[image_index];

				char labelpath_dif[4096];
				find_replace(path_dif, "images", "labels", labelpath_dif);
				find_replace(labelpath_dif, "JPEGImages", "labels", labelpath_dif);
				find_replace(labelpath_dif, ".jpg", ".txt", labelpath_dif);
				find_replace(labelpath_dif, ".JPEG", ".txt", labelpath_dif);
				find_replace(labelpath_dif, ".png", ".txt", labelpath_dif);				
				truth_dif = read_boxes(labelpath_dif, &num_labels_dif);
			}

			for (i = 0; i < nboxes; ++i) {

				int class_id;
				for (class_id = 0; class_id < classes; ++class_id) {
					float prob = dets[i].prob[class_id];
					if (prob > 0) {
						detections_count++;
						detections = realloc(detections, detections_count * sizeof(box_prob));
						detections[detections_count - 1].b = dets[i].bbox;
						detections[detections_count - 1].p = prob;
						detections[detections_count - 1].image_index = image_index;
						detections[detections_count - 1].class_id = class_id;
						detections[detections_count - 1].truth_flag = 0;
						detections[detections_count - 1].unique_truth_index = -1;

						int truth_index = -1;
						float max_iou = 0;
						for (j = 0; j < num_labels; ++j)
						{
							box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
							//printf(" IoU = %f, prob = %f, class_id = %d, truth[j].id = %d \n", 
							//	box_iou(dets[i].bbox, t), prob, class_id, truth[j].id);
							float current_iou = box_iou(dets[i].bbox, t);
							if (current_iou > iou_thresh && class_id == truth[j].id) {
								if (current_iou > max_iou) {
									max_iou = current_iou;
									truth_index = unique_truth_count + j;
								}
							}
						}

						// best IoU
						if (truth_index > -1) {
							detections[detections_count - 1].truth_flag = 1;
							detections[detections_count - 1].unique_truth_index = truth_index;
						}
						else {
							// if object is difficult then remove detection
							for (j = 0; j < num_labels_dif; ++j) {
								box t = { truth_dif[j].x, truth_dif[j].y, truth_dif[j].w, truth_dif[j].h };
								float current_iou = box_iou(dets[i].bbox, t);
								if (current_iou > iou_thresh && class_id == truth_dif[j].id) {
									--detections_count;
									break;
								}
							}
						}

						// calc avg IoU, true-positives, false-positives for required Threshold
						if (prob > thresh_calc_avg_iou) {
							if (truth_index > -1) {
								avg_iou += max_iou;
								++tp_for_thresh;
							}
							else
								fp_for_thresh++;
						}
					}
				}
			}
			
			unique_truth_count += num_labels;

			free_detections(dets, nboxes);
			free(id);
			free_image(val[t]);
			free_image(val_resized[t]);
		}
	}

	avg_iou = avg_iou / (tp_for_thresh + fp_for_thresh);

	
	// SORT(detections)
	qsort(detections, detections_count, sizeof(box_prob), detections_comparator);
	
	typedef struct {
		double precision;
		double recall;
		int tp, fp, fn;
	} pr_t;

	// for PR-curve
	pr_t **pr = calloc(classes, sizeof(pr_t*));
	for (i = 0; i < classes; ++i) {
		pr[i] = calloc(detections_count, sizeof(pr_t));
	}
	printf("detections_count = %d, unique_truth_count = %d  \n", detections_count, unique_truth_count);


	int *truth_flags = calloc(unique_truth_count, sizeof(int));

	int rank;
	for (rank = 0; rank < detections_count; ++rank) {
		if(rank % 100 == 0)
			printf(" rank = %d of ranks = %d \r", rank, detections_count);

		if (rank > 0) {
			int class_id;
			for (class_id = 0; class_id < classes; ++class_id) {
				pr[class_id][rank].tp = pr[class_id][rank - 1].tp;
				pr[class_id][rank].fp = pr[class_id][rank - 1].fp;
			}
		}

		box_prob d = detections[rank];
		// if (detected && isn't detected before)
		if (d.truth_flag == 1) {
			if (truth_flags[d.unique_truth_index] == 0) 
			{
				truth_flags[d.unique_truth_index] = 1;
				pr[d.class_id][rank].tp++;	// true-positive
			}
		}
		else {
			pr[d.class_id][rank].fp++;	// false-positive
		}

		for (i = 0; i < classes; ++i) 
		{
			const int tp = pr[i][rank].tp;
			const int fp = pr[i][rank].fp;
			const int fn = truth_classes_count[i] - tp;	// false-negative = objects - true-positive
			pr[i][rank].fn = fn;

			if ((tp + fp) > 0) pr[i][rank].precision = (double)tp / (double)(tp + fp);
			else pr[i][rank].precision = 0;

			if ((tp + fn) > 0) pr[i][rank].recall = (double)tp / (double)(tp + fn);
			else pr[i][rank].recall = 0;
		}
	}

	free(truth_flags);
	
	
	double mean_average_precision = 0;

	for (i = 0; i < classes; ++i) {
		double avg_precision = 0;
		int point;
		for (point = 0; point < 11; ++point) {
			double cur_recall = point * 0.1;
			double cur_precision = 0;
			for (rank = 0; rank < detections_count; ++rank)
			{
				if (pr[i][rank].recall >= cur_recall) {	// > or >=
					if (pr[i][rank].precision > cur_precision) {
						cur_precision = pr[i][rank].precision;
					}
				}
			}
			//printf("class_id = %d, point = %d, cur_recall = %.4f, cur_precision = %.4f \n", i, point, cur_recall, cur_precision);

			avg_precision += cur_precision;
		}
		avg_precision = avg_precision / 11;
		printf("class_id = %d, name = %s, \t ap = %2.2f %% \n", i, names[i], avg_precision*100);
		mean_average_precision += avg_precision;
	}
	
	const float cur_precision = (float)tp_for_thresh / ((float)tp_for_thresh + (float)fp_for_thresh);
	const float cur_recall = (float)tp_for_thresh / ((float)tp_for_thresh + (float)(unique_truth_count - tp_for_thresh));
	const float f1_score = 2.F * cur_precision * cur_recall / (cur_precision + cur_recall);
	printf(" for thresh = %1.2f, precision = %1.2f, recall = %1.2f, F1-score = %1.2f \n",
		thresh_calc_avg_iou, cur_precision, cur_recall, f1_score);

	printf(" for thresh = %0.2f, TP = %d, FP = %d, FN = %d, average IoU = %2.2f %% \n", 
		thresh_calc_avg_iou, tp_for_thresh, fp_for_thresh, unique_truth_count - tp_for_thresh, avg_iou * 100);

	mean_average_precision = mean_average_precision / classes;
	printf("\n mean average precision (mAP) = %f, or %2.2f %% \n", mean_average_precision, mean_average_precision*100);


	for (i = 0; i < classes; ++i) {
		free(pr[i]);
	}
	free(pr);
	free(detections);
	free(truth_classes_count);

	fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}

#ifdef OPENCV
typedef struct {
	float w, h;
} anchors_t;

int anchors_comparator(const void *pa, const void *pb)
{
	anchors_t a = *(anchors_t *)pa;
	anchors_t b = *(anchors_t *)pb;
	float diff = b.w - a.w;
	if (diff < 0) return 1;
	else if (diff > 0) return -1;
	return 0;
}

void calc_anchors(char *datacfg, int num_of_clusters, int width, int height, int show)
{
	printf("\n num_of_clusters = %d, width = %d, height = %d \n", num_of_clusters, width, height);

	//float pointsdata[] = { 1,1, 2,2, 6,6, 5,5, 10,10 };
	float *rel_width_height_array = calloc(1000, sizeof(float));

	list *options = read_data_cfg(datacfg);
	char *train_images = option_find_str(options, "train", "data/train.list");
	list *plist = get_paths(train_images);
	int number_of_images = plist->size;
	char **paths = (char **)list_to_array(plist);

	int number_of_boxes = 0;
	printf(" read labels from %d images \n", number_of_images);

	int i, j;
	for (i = 0; i < number_of_images; ++i) {
		char *path = paths[i];
		char labelpath[4096];
		find_replace(path, "images", "labels", labelpath);
		find_replace(labelpath, "JPEGImages", "labels", labelpath);
		find_replace(labelpath, ".jpg", ".txt", labelpath);
		find_replace(labelpath, ".JPEG", ".txt", labelpath);
		find_replace(labelpath, ".png", ".txt", labelpath);
		int num_labels = 0;
		box_label *truth = read_boxes(labelpath, &num_labels);
		//printf(" new path: %s \n", labelpath);
		for (j = 0; j < num_labels; ++j)
		{
			number_of_boxes++;
			rel_width_height_array = realloc(rel_width_height_array, 2 * number_of_boxes * sizeof(float));
			rel_width_height_array[number_of_boxes * 2 - 2] = truth[j].w * width;
			rel_width_height_array[number_of_boxes * 2 - 1] = truth[j].h * height;
			printf("\r loaded \t image: %d \t box: %d", i+1, number_of_boxes);
		}
	}
	printf("\n all loaded. \n");

	CvMat* points = cvCreateMat(number_of_boxes, 2, CV_32FC1);
	CvMat* centers = cvCreateMat(num_of_clusters, 2, CV_32FC1);
	CvMat* labels = cvCreateMat(number_of_boxes, 1, CV_32SC1);

	for (i = 0; i < number_of_boxes; ++i) {
		points->data.fl[i * 2] = rel_width_height_array[i * 2];
		points->data.fl[i * 2 + 1] = rel_width_height_array[i * 2 + 1];
		//cvSet1D(points, i * 2, cvScalar(rel_width_height_array[i * 2], 0, 0, 0));
		//cvSet1D(points, i * 2 + 1, cvScalar(rel_width_height_array[i * 2 + 1], 0, 0, 0));
	}


	const int attemps = 10;
	double compactness;

	enum {
		KMEANS_RANDOM_CENTERS = 0,
		KMEANS_USE_INITIAL_LABELS = 1,
		KMEANS_PP_CENTERS = 2
	};
	
	printf("\n calculating k-means++ ...");
	// Should be used: distance(box, centroid) = 1 - IoU(box, centroid)
	cvKMeans2(points, num_of_clusters, labels, 
		cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10000, 0), attemps, 
		0, KMEANS_PP_CENTERS,
		centers, &compactness);

	// sort anchors
	qsort(centers->data.fl, num_of_clusters, 2*sizeof(float), anchors_comparator);

	//orig 2.0 anchors = 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52
	//float orig_anch[] = { 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52 };
	// worse than ours (even for 19x19 final size - for input size 608x608)

	//orig anchors = 1.3221,1.73145, 3.19275,4.00944, 5.05587,8.09892, 9.47112,4.84053, 11.2364,10.0071
	//float orig_anch[] = { 1.3221,1.73145, 3.19275,4.00944, 5.05587,8.09892, 9.47112,4.84053, 11.2364,10.0071 };
	// orig (IoU=59.90%) better than ours (59.75%)

	//gen_anchors.py = 1.19, 1.99, 2.79, 4.60, 4.53, 8.92, 8.06, 5.29, 10.32, 10.66
	//float orig_anch[] = { 1.19, 1.99, 2.79, 4.60, 4.53, 8.92, 8.06, 5.29, 10.32, 10.66 };

	// ours: anchors = 9.3813,6.0095, 3.3999,5.3505, 10.9476,11.1992, 5.0161,9.8314, 1.5003,2.1595
	//float orig_anch[] = { 9.3813,6.0095, 3.3999,5.3505, 10.9476,11.1992, 5.0161,9.8314, 1.5003,2.1595 };
	//for (i = 0; i < num_of_clusters * 2; ++i) centers->data.fl[i] = orig_anch[i];
	
	//for (i = 0; i < number_of_boxes; ++i)
	//	printf("%2.2f,%2.2f, ", points->data.fl[i * 2], points->data.fl[i * 2 + 1]);

	float avg_iou = 0;
	for (i = 0; i < number_of_boxes; ++i) {
		float box_w = points->data.fl[i * 2];
		float box_h = points->data.fl[i * 2 + 1];
		//int cluster_idx = labels->data.i[i];		
		int cluster_idx = 0;
		float min_dist = FLT_MAX;
		for (j = 0; j < num_of_clusters; ++j) {
			float anchor_w = centers->data.fl[j * 2];
			float anchor_h = centers->data.fl[j * 2 + 1];
			float w_diff = anchor_w - box_w;
			float h_diff = anchor_h - box_h;
			float distance = sqrt(w_diff*w_diff + h_diff*h_diff);
			if (distance < min_dist) min_dist = distance, cluster_idx = j;
		}
		
		float anchor_w = centers->data.fl[cluster_idx * 2];
		float anchor_h = centers->data.fl[cluster_idx * 2 + 1];
		float min_w = (box_w < anchor_w) ? box_w : anchor_w;
		float min_h = (box_h < anchor_h) ? box_h : anchor_h;
		float box_intersect = min_w*min_h;
		float box_union = box_w*box_h + anchor_w*anchor_h - box_intersect;
		float iou = box_intersect / box_union;
		if (iou > 1 || iou < 0) {
			printf(" i = %d, box_w = %d, box_h = %d, anchor_w = %d, anchor_h = %d, iou = %f \n",
				i, box_w, box_h, anchor_w, anchor_h, iou);
		}
		else avg_iou += iou;
	}
	avg_iou = 100 * avg_iou / number_of_boxes;
	printf("\n avg IoU = %2.2f %% \n", avg_iou);

	char buff[1024];
	FILE* fw = fopen("anchors.txt", "wb");
	printf("\nSaving anchors to the file: anchors.txt \n");
	printf("anchors = ");
	for (i = 0; i < num_of_clusters; ++i) {
		sprintf(buff, "%2.4f,%2.4f", centers->data.fl[i * 2], centers->data.fl[i * 2 + 1]);
		printf("%s", buff);
		fwrite(buff, sizeof(char), strlen(buff), fw);
		if (i + 1 < num_of_clusters) {
			fwrite(", ", sizeof(char), 2, fw);
			printf(", ");
		}
	}
	printf("\n");
	fclose(fw);

	if (show) {
		size_t img_size = 700;
		IplImage* img = cvCreateImage(cvSize(img_size, img_size), 8, 3);
		cvZero(img);
		for (j = 0; j < num_of_clusters; ++j) {
			CvPoint pt1, pt2;
			pt1.x = pt1.y = 0;
			pt2.x = centers->data.fl[j * 2] * img_size / width;
			pt2.y = centers->data.fl[j * 2 + 1] * img_size / height;
			cvRectangle(img, pt1, pt2, CV_RGB(255, 255, 255), 1, 8, 0);
		}

		for (i = 0; i < number_of_boxes; ++i) {
			CvPoint pt;
			pt.x = points->data.fl[i * 2] * img_size / width;
			pt.y = points->data.fl[i * 2 + 1] * img_size / height;
			int cluster_idx = labels->data.i[i];
			int red_id = (cluster_idx * (uint64_t)123 + 55) % 255;
			int green_id = (cluster_idx * (uint64_t)321 + 33) % 255;
			int blue_id = (cluster_idx * (uint64_t)11 + 99) % 255;
			cvCircle(img, pt, 1, CV_RGB(red_id, green_id, blue_id), CV_FILLED, 8, 0);
			//if(pt.x > img_size || pt.y > img_size) printf("\n pt.x = %d, pt.y = %d \n", pt.x, pt.y);
		}
		cvShowImage("clusters", img);
		cvWaitKey(0);
		cvReleaseImage(&img);
		cvDestroyAllWindows();
	}

	free(rel_width_height_array);
	cvReleaseMat(&points);
	cvReleaseMat(&centers);
	cvReleaseMat(&labels);
}
#else
void calc_anchors(char *datacfg, int num_of_clusters, int width, int height, int show) {
	printf(" k-means++ can't be used without OpenCV, because there is used cvKMeans2 implementation \n");
}
#endif // OPENCV

void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, int dont_show)
{
	dont_show = 0;
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

	image **alphabet = NULL;// load_alphabet();
    network net = parse_network_cfg_custom(cfgfile, 1);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.45;	// 0.4F

	char   *files[] = { "E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00001.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00004.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00008.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00012.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00019.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00027.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00033.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00034.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00038.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00040.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00041.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00049.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00050.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00051.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00054.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00059.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00067.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00070.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00083.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00085.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00090.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00101.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00103.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00104.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00117.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00128.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00135.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00154.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00167.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00168.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00169.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00170.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00171.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00175.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00178.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00191.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00192.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00200.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00202.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00208.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00212.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00214.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00215.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00217.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00220.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00229.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00236.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00240.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00241.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00245.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00246.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00249.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00253.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00254.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00260.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00261.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00265.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00270.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00282.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00285.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00287.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00289.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00293.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00294.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00296.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00298.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00299.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00306.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00312.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00313.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00324.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00334.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00338.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00352.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00354.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00355.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00356.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00368.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00379.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00381.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00385.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00389.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00390.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00393.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00397.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00400.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00405.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00407.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00411.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00412.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00415.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00420.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00427.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00432.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00433.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00444.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00445.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00448.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00450.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00453.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00454.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00459.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00460.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00464.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00469.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00473.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00474.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00478.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00487.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00495.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00502.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00503.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00506.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00514.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00516.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00520.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00526.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00527.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00535.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00536.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00538.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00542.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00551.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00554.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00557.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00561.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00563.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00575.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00577.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00579.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00586.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00588.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00590.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00595.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00596.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00597.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00599.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00603.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00605.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00606.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00609.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00611.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00616.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00620.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00622.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00626.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00627.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00633.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00634.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00635.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00638.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00642.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00645.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00646.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00648.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00666.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00667.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00669.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00672.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00673.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00674.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00688.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00693.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00694.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00699.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00701.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00723.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00726.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00729.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00733.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00742.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00746.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00750.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00751.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00760.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00766.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00770.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00774.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00790.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00792.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00796.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00797.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00799.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00800.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00801.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00803.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00804.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00805.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00808.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00811.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00812.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00825.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00834.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00836.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00838.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00839.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00840.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00848.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00851.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00857.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00860.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00866.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00872.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00874.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00877.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00880.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00882.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00883.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00884.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00889.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00890.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00900.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00904.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00906.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00911.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00912.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00913.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00915.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00921.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00924.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00926.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00930.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00934.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00936.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00942.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00944.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00949.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00952.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00953.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00956.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00960.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00962.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00967.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00975.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00978.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00979.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00983.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00991.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00993.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/00997.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01000.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01001.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01009.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01010.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01024.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01035.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01038.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01040.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01048.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01052.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01055.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01061.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01062.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01070.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01073.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01077.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01081.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01084.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01086.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01091.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01094.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01098.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01105.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01114.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01117.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01122.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01123.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01126.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01131.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01137.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01138.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01144.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01148.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01150.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01153.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01154.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01156.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01158.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01159.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01160.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01166.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01174.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01180.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01181.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01183.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01184.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01187.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01189.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01190.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01194.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01198.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01205.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01206.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01208.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01209.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01216.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01218.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01219.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01227.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01228.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01230.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01232.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01233.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01237.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01238.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01240.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01247.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01248.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01253.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01254.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01258.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01260.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01265.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01276.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01278.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01290.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01299.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01309.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01310.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01315.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01320.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01324.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01327.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01331.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01334.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01335.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01336.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01337.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01340.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01342.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01344.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01346.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01350.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01354.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01362.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01372.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01381.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01385.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01386.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01387.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01389.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01390.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01395.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01401.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01412.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01421.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01422.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01426.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01429.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01433.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01440.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01445.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01448.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01450.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01453.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01458.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01461.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01467.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01472.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01473.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01478.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01479.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01481.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01483.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01485.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01489.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01491.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01492.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01495.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01496.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01499.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01501.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01504.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01508.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01511.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01512.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01517.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01519.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01520.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01522.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01525.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01530.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01532.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01534.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01542.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01552.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01554.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01555.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01558.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01561.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01563.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01567.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01570.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01574.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01576.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01578.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01580.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01584.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01585.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01591.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01595.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01597.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01599.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01604.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01608.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01615.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01622.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01626.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01628.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01631.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01635.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01640.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01653.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01654.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01657.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01664.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01665.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01667.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01669.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01671.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01674.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01677.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01678.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01679.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01680.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01684.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01687.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01690.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01692.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01693.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01700.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01706.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01715.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01720.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01737.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01742.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01744.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01751.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01755.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01759.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01765.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01767.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01771.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01782.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01783.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01786.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01790.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01793.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01794.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01798.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01801.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01803.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01808.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01810.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01817.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01818.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01833.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01841.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01842.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01845.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01849.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01853.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01858.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01859.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01860.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01866.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01873.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01880.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01883.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01892.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01894.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01897.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01901.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01903.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01913.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01915.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01916.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01918.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01924.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01928.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01930.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01932.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01935.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01941.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01952.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01953.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01958.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01961.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01963.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01970.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/01984.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02003.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02004.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02005.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02010.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02016.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02019.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02020.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02021.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02022.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02023.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02024.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02026.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02031.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02032.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02036.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02037.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02038.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02041.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02045.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02048.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02049.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02053.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02057.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02062.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02064.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02065.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02069.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02073.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02076.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02077.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02081.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02084.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02090.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02095.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02098.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02103.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02109.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02120.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02127.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02131.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02134.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02136.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02138.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02142.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02144.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02145.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02147.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02154.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02157.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02159.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02160.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02164.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02172.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02177.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02182.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02184.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02191.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02195.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02196.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02197.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02202.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02204.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02206.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02212.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02213.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02214.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02219.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02222.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02224.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02226.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02230.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02235.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02238.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02239.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02241.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02242.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02251.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02254.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02255.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02280.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02286.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02290.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02291.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02294.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02300.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02307.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02317.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02318.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02322.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02335.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02337.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02340.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02345.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02361.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02364.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02367.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02368.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02372.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02373.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02375.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02382.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02389.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02391.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02392.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02402.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02404.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02405.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02406.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02410.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02416.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02421.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02429.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02434.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02448.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02452.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02455.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02457.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02460.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02462.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02471.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02482.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02492.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02496.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02500.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02509.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02514.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02518.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02519.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02521.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02523.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02524.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02527.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02528.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02532.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02534.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02535.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02537.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02538.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02540.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02542.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02543.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02545.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02555.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02556.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02558.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02560.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02562.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02564.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02577.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02581.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02585.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02591.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02592.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02595.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02605.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02608.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02609.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02610.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02612.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02621.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02623.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02628.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02629.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02630.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02632.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02634.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02637.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02638.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02640.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02643.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02649.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02654.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02655.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02659.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02667.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02673.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02675.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02678.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02681.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02688.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02689.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02690.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02691.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02694.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02697.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02698.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02699.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02707.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02718.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02728.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02743.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02753.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02754.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02757.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02763.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02771.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02779.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02787.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02788.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02793.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02794.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02795.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02801.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02802.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02803.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02807.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02809.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02813.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02814.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02820.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02823.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02827.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02829.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02831.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02840.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02842.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02844.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02846.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02849.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02853.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02859.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02863.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02864.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02865.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02867.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02872.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02876.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02881.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02883.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02886.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02895.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02898.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02902.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02907.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02911.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02916.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02917.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02924.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02927.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02928.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02931.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02932.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02937.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02943.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02953.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02955.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02960.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02963.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02965.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02966.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02968.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02969.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02971.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02972.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02976.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02977.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02978.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02979.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02982.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02985.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02990.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02993.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/02995.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03000.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03001.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03004.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03007.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03009.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03011.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03014.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03015.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03024.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03028.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03032.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03033.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03037.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03046.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03054.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03055.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03056.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03057.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03062.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03064.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03066.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03069.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03083.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03088.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03101.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03103.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03109.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03112.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03116.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03123.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03124.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03127.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03128.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03132.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03133.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03138.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03145.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03146.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03153.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03156.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03157.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03159.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03160.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03161.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03162.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03165.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03169.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03170.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03178.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03179.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03183.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03187.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03188.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03195.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03200.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03202.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03204.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03206.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03208.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03215.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03217.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03225.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03232.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03236.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03247.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03252.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03253.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03255.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03261.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03262.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03263.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03266.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03267.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03271.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03283.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03284.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03286.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03292.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03293.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03294.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03295.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03304.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03306.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03307.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03311.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03314.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03315.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03317.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03318.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03325.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03326.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03327.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03329.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03330.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03338.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03343.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03367.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03369.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03376.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03378.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03381.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03383.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03386.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03387.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03395.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03405.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03411.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03417.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03418.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03421.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03429.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03435.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03447.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03452.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03454.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03455.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03459.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03467.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03471.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03475.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03477.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03478.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03481.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03483.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03486.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03489.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03491.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03493.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03498.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03505.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03509.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03510.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03513.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03537.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03538.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03541.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03542.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03543.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03550.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03552.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03554.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03557.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03560.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03570.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03571.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03576.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03577.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03578.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03579.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03581.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03586.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03592.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03597.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03602.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03607.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03611.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03615.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03616.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03617.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03626.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03634.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03636.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03638.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03639.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03642.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03643.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03646.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03647.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03648.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03649.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03662.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03664.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03665.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03668.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03670.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03674.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03675.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03676.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03677.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03680.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03682.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03684.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03691.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03699.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03702.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03706.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03708.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03711.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03722.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03731.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03736.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03739.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03740.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03742.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03746.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03755.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03761.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03764.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03765.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03769.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03775.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03779.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03780.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03782.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03785.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03786.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03787.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03789.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03795.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03796.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03799.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03806.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03808.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03811.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03814.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03818.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03820.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03821.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03822.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03827.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03835.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03837.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03838.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03841.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03842.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03852.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03853.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03856.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03859.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03864.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03868.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03869.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03872.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03873.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03877.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03882.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03892.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03902.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03908.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03909.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03910.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03911.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03917.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03923.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03924.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03925.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03927.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03938.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03939.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03944.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03946.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03947.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03948.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03957.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03964.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03966.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03971.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03973.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03985.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/03990.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04003.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04004.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04006.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04010.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04014.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04023.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04029.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04031.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04033.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04038.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04040.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04052.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04059.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04060.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04062.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04065.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04074.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04077.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04078.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04082.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04085.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04087.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04091.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04094.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04097.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04100.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04102.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04103.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04109.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04113.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04115.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04118.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04124.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04126.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04128.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04131.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04132.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04135.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04139.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04148.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04150.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04151.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04153.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04162.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04170.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04173.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04176.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04177.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04184.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04186.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04188.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04196.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04198.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04201.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04202.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04205.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04211.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04212.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04224.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04227.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04228.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04231.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04232.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04234.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04241.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04244.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04245.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04253.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04254.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04255.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04262.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04263.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04264.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04265.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04271.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04274.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04277.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04278.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04282.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04283.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04287.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04290.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04291.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04294.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04301.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04305.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04306.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04308.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04312.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04313.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04315.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04322.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04325.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04336.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04340.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04342.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04345.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04352.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04353.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04354.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04363.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04372.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04374.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04378.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04380.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04382.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04383.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04386.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04391.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04406.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04420.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04430.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04434.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04444.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04451.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04453.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04456.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04457.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04464.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04465.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04467.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04478.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04487.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04499.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04500.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04501.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04502.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04508.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04509.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04513.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04515.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04518.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04525.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04530.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04533.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04536.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04547.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04558.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04560.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04562.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04575.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04578.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04583.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04585.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04590.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04594.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04601.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04602.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04609.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04612.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04616.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04618.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04621.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04625.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04634.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04638.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04643.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04647.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04653.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04664.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04679.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04687.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04693.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04694.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04698.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04706.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04707.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04710.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04720.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04721.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04723.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04724.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04726.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04731.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04732.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04743.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04746.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04748.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04751.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04752.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04759.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04760.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04761.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04764.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04766.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04770.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04772.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04781.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04789.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04801.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04809.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04811.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04812.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04816.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04817.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04821.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04825.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04828.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04832.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04841.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04844.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04845.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04850.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04851.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04853.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04856.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04862.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04865.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04868.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04870.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04874.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04879.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04884.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04885.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04894.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04901.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04903.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04907.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04912.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04913.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04916.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04924.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04935.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04941.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04942.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04948.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04949.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04953.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04959.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04960.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04961.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04965.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04972.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04975.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04984.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/04997.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05000.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05011.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05015.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05019.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05022.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05028.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05029.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05030.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05035.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05036.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05044.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05050.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05052.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05055.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05057.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05060.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05067.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05068.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05071.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05072.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05077.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05083.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05092.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05098.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05103.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05104.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05111.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05127.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05129.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05131.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05132.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05133.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05134.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05138.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05139.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05143.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05146.jpg",
		"E:/github/darknet_windows/build/darknet/x64/data/voc/VOCdevkit/VOC2007/JPEGImages/05153.jpg"
	};

	int   i = 0;

	while (strcmp(files[i], " ") != 0) {

		if (files[i]) {
			strncpy(input, files[i], 256);
			if (input[strlen(input) - 1] == 0x0d) input[strlen(input) - 1] = 0;
		}
		else {
			printf("Enter Image Path: ");
			fflush(stdout);
			input = fgets(input, 256, stdin);
			if (!input) return;
			strtok(input, "\n");
		}
		image im = load_image_color(input, 0, 0);
		int letterbox = 0;
		image sized = resize_image(im, net.w, net.h);
		//image sized = letterbox_image(im, net.w, net.h); letterbox = 1;
		layer l = net.layers[net.n - 1];

		//box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
		//float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
		//for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));

		float *X = sized.data;
		time = clock();
		network_predict(net, X);
		printf("%s: Predicted in %f seconds.\n", input, sec(clock() - time));
		//get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0);
		// if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
		//draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);
		int nboxes = 0;
		detection *dets = get_network_boxes(&net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letterbox);
		if (nms) do_nms_sort_v3(dets, nboxes, l.classes, nms);
		printf("nboxes %d \n", nboxes);
		draw_detections_v3(im, dets, nboxes, thresh, names, alphabet, l.classes);
		free_detections(dets, nboxes);
		save_image(im, "predictions");
		if (!dont_show) {
			show_image(im, "predictions");
		}

		free_image(im);
		free_image(sized);
		//free(boxes);
		//free_ptrs((void **)probs, l.w*l.h*l.n);
#ifdef OPENCV
		if (!dont_show) {
			cvWaitKey(0);
			//cvDestroyAllWindows();
		}
#endif
		//if (filename) break;
		i++;
		//puts(test[i++]);

	}
		
//    while(1){
//        if(filename){
//            strncpy(input, filename, 256);
//			if (input[strlen(input) - 1] == 0x0d) input[strlen(input) - 1] = 0;
//        } else {
//            printf("Enter Image Path: ");
//            fflush(stdout);
//            input = fgets(input, 256, stdin);
//            if(!input) return;
//            strtok(input, "\n");
//        }
//        image im = load_image_color(input,0,0);
//		int letterbox = 0;
//        image sized = resize_image(im, net.w, net.h);
//		//image sized = letterbox_image(im, net.w, net.h); letterbox = 1;
//        layer l = net.layers[net.n-1];
//
//        //box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
//        //float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
//        //for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
//
//        float *X = sized.data;
//        time=clock();
//        network_predict(net, X);
//        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
//        //get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0);
//		// if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
//		//draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);
//		int nboxes = 0;
//		detection *dets = get_network_boxes(&net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letterbox);
//		if (nms) do_nms_sort_v3(dets, nboxes, l.classes, nms);
//		printf("nboxes %d \n", nboxes);
//		draw_detections_v3(im, dets, nboxes, thresh, names, alphabet, l.classes);
//		free_detections(dets, nboxes);
//        save_image(im, "predictions");
//		if (!dont_show) {
//			show_image(im, "predictions");
//		}
//
//        free_image(im);
//        free_image(sized);
//        //free(boxes);
//        //free_ptrs((void **)probs, l.w*l.h*l.n);
//#ifdef OPENCV
//		if (!dont_show) {
//			cvWaitKey(0);
//			cvDestroyAllWindows();
//		}
//#endif
//        if (filename) break;
//    }
}

void run_detector(int argc, char **argv)
{
	int dont_show = find_arg(argc, argv, "-dont_show");
	int show = find_arg(argc, argv, "-show");
	int http_stream_port = find_int_arg(argc, argv, "-http_port", -1);
	char *out_filename = find_char_arg(argc, argv, "-out_filename", 0);
	char *outfile = find_char_arg(argc, argv, "-out", 0);
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .25);	// 0.24
	float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
	int num_of_clusters = find_int_arg(argc, argv, "-num_of_clusters", 5);
	int width = find_int_arg(argc, argv, "-width", 13);
	int heigh = find_int_arg(argc, argv, "-heigh", 13);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear");

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
	if(weights)
		if (weights[strlen(weights) - 1] == 0x0d) weights[strlen(weights) - 1] = 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    if(0==strcmp(argv[2], "test")) test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh, dont_show);
    else if(0==strcmp(argv[2], "train")) train_detector(datacfg, cfg, weights, gpus, ngpus, clear, dont_show);
    else if(0==strcmp(argv[2], "valid")) validate_detector(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "recall")) validate_detector_recall(datacfg, cfg, weights);
	else if(0==strcmp(argv[2], "map")) validate_detector_map(datacfg, cfg, weights, thresh);
	else if(0==strcmp(argv[2], "calc_anchors")) calc_anchors(datacfg, num_of_clusters, width, heigh, show);
    else if(0==strcmp(argv[2], "demo")) {
        list *options = read_data_cfg(datacfg);
        int classes = option_find_int(options, "classes", 20);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
		if(filename)
			if (filename[strlen(filename) - 1] == 0x0d) filename[strlen(filename) - 1] = 0;
        demo(cfg, weights, thresh, hier_thresh, cam_index, filename, names, classes, frame_skip, prefix, out_filename,
			http_stream_port, dont_show);
    }
}
