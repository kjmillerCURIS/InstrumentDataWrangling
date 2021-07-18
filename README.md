Codes for downloading and preparing surgical instrument data, which I intend to use for multiple side projects.



Example commands below assume data is being stored in existing "~/InstrumentData" directory.



Download videos:

    python download_videos.py VideoWranglingParams ~/InstrumentData/videos ~/InstrumentData/video_hash_dict.pkl



Extract frames:

    python extract_frames.py ~/InstrumentData/videos VideoWranglingParams ~/InstrumentData/frames



Pick random subsample from the frames:

    python pick_random_sample.py ~/InstrumentData/frames 100 0 ~/InstrumentData/random_sample_frames_100_0



Gather labelled bboxes into one pkl file:

    python make_bboxes_dict.py ~/InstrumentData/random_sample_frames_100_0 ~/InstrumentData/relevance_gt_label_JSONs ~/InstrumentData/relevance_gt_bboxes_dict.pkl



Train a model (one example):

    python relevance_training.py ~/InstrumentData/random_sample_frames_100_0 ~/InstrumentData/relevance_gt_bboxes_dict.pkl RelevanceParams16Comp1Trust ~/InstrumentData/relevance_results/RelevanceParams16Comp1Trust-model.pt ~/InstrumentData/relevance_results/RelevanceParams16Comp1Trust-result.pkl



Visualize segmentations:

    python explore_segmentations.py ~/InstrumentData/frames






TODO:

1.) Items (f.) and bbox-vis part of (g.). And also choose the best relevance model. And also write a script that bulk-crops the entire dataset.




Relevance Detector:

Lots of frames are either irrelevant or only have a small part that is relevant. It would be nice to be able to train a model to extract the relevant regions, using as little supervision as possible. Ideally I would label, train, visualize detections on a subset of the data, add more labels, retrain, visualize more detections, etc.

In order to be as data-efficient as possible, let's do the following:

a.) Pick a subset of the images to supervise. Label any relevant regions with bounding boxes.

b.) Put each image through a pretrained ResNet50. Try very hard to keep all resizing isotropic. Hopefully we can just downsize the image so its smaller dimension is something like 224, and then the ResNet50 will give us an output of roughly the same aspect ratio, something like round(7w/h) x 7.

c.) For each pixel of the output, see how much bounding-box is contained within its receptive field. We'll pretend that all output pixels have the same size and shape receptive field. Store this proportion.

d.) Train a classifier that takes the output pixel embedding and tries to predict the label. The label will be decided randomly on-the-fly based on the proportion. We'll store a dictionary mapping key to embedding, and another one mapping key to proportion. We'll maintain lists of keys that are completely 0 or 1 in proportion, and keys that are in-between. We'll randomly assign all the in-between keys (based on their proportion) every time we sample a batch. Combine the assigned in-between keys with the 0 and 1 keys and sample a batch with equal amount positives and negatives. The random assignment is efficient because it doesn't move around the embeddings, just the keys. Validation will also use this random assignment so that it's easy to compute a balanced validation loss. Reassignment will happen every time validation loss is computed. Finally, there's augmentation: do it for training, not for validation. The augmentations will be vertical and horizontal flipping and 90-degree rotations, and we'll just do every possible augmentation at the beginning so we don't have to recompute the embeddings.

e.) Classifier will be sigmoid(max_k(w_k * x + b_k)). len(x) == 2048, but I think early-stopping might save us even if there's fewer samples than parameters. On the other hand, I'd like to learn how to do regularization in PyTorch. So I'll constrain L2(w_k) <= C for all k, or something like that. And I'll do my first experiment with very weak/no regularization. I like weight clipping better than other regularizations because the max() will create lots of zero gradients, and I don't want my weights to decay away. C can be some multiple of the average L2 norm of the ResNet50 readout classifiers.

f.) This classifier will get us a really coarse segmentation map. But we need bounding boxes! How to get them? Well, we'll keep it simple. Find the connected components. Get tight boxes around them. Filter by width, height, and compactness. Do this for the original map, an opened map, and a closed map, and combine all the valid bboxes (edit: let's look at the actual segmentations before deciding that).

g.) Make some kind of GUI that lets you quickly shuffle between different models and different bboxification choices on the same random set of test images, and also has a choice for resampling those test images. Once you're happy, run your favorite pipeline on the entire dataset and save a crop from each bbox. Oh, and we also want a GUI that just looks at segmentations.
