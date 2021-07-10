Codes for downloading and preparing surgical instrument data, which I intend to use for multiple side projects.



Download videos:

    python download_videos.py <params_key> <video_dir> <hash_dict_filename>



Extract frames:

    python extract_frames.py videos VideoWranglingParams frames



Pick random subsample from the frames:

    python pick_random_sample.py frames 100 0 random_sample_frames_100_0



TODO:

1.) Work on Relevance Detector (see below - done with (a.)).




Relevance Detector:

Lots of frames are either irrelevant or only have a small part that is relevant. It would be nice to be able to train a model to extract the relevant regions, using as little supervision as possible. Ideally I would label, train, visualize detections on a subset of the data, add more labels, retrain, visualize more detections, etc.

In order to be as data-efficient as possible, let's do the following:

a.) Pick a subset of the images to supervise. Label any relevant regions with bounding boxes.

b.) Put each image through a pretrained ResNet50. Try very hard to keep all resizing isotropic. Hopefully we can just downsize the image so its smaller dimension is something like 224, and then the ResNet50 will give us an output of roughly the same aspect ratio, something like round(7w/h) x 7.

c.) For each pixel of the output, see how much bounding-box is contained within its receptive field. We'll pretend that all output pixels have the same size and shape receptive field. Store this proportion.

d.) Train a classifier that takes the output pixel embedding and tries to predict the label. The label will be decided randomly on-the-fly based on the proportion. We'll store a dictionary mapping key to embedding, and another one mapping key to proportion. We'll maintain lists of keys that are completely 0 or 1 in proportion, and keys that are in-between. We'll randomly assign all the in-between keys (based on their proportion) every time we sample a batch. Combine the assigned in-between keys with the 0 and 1 keys and sample a batch with equal amount positives and negatives. The random assignment is efficient because it doesn't move around the embeddings, just the keys.

e.) Classifier will be sigmoid(max_k(w_k * x + b_k)). len(x) == 2048, but I think early-stopping might save us even if there's fewer samples than parameters. On the other hand, I'd like to learn how to do regularization in PyTorch. So I'll constrain L2(w_k) <= C for all k, or something like that. And I'll do my first experiment with very weak/no regularization. I like weight clipping better than other regularizations because the max() will create lots of zero gradients, and I don't want my weights to decay away! Also, I can look at the top layer of the pretrained model to figure out a good guess for C. I'll just use some off-the-shelf PyTorch thing to initialize (need to learn about that too!).

f.) This classifier will get us a really coarse segmentation map. But we need bounnding boxes! How to get them? Well, we'll keep it simple. Find the connected components. Get tight boxes around them. Filter by width, height, and compactness. Do this for the original map, an opened map, and a closed map, and combine all the valid bboxes.

g.) Make some kind of GUI that lets you quickly shuffle between different models and different bboxification choices on the same random set of test images, and also has a choice for resampling those test images. Once you're happy, run your favorite pipeline on the entire dataset and save a crop from each bbox.
