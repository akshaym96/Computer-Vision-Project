## Yelp Photo Classification Task

We need to predict the labels describing the business attributes of restaurants from the user submitted pictures. This is a Multi-Instance Multi-Label (MIML) classification problem. Each photo is assigned a business id and each business id is assigned multiple labels as given below.
The problem was launched as Kaggle competition. [10]

Insert image here

The original dataset contains nearly 2.3 lakh images taken by the users. Each photo is assigned a id called photoID. Each photoID is mapped to a businessID and each of the businessID is assigned multiple labels as shown in the figure above.

The different labels are listed below:
0: good_for_lunch
1: good_for_dinner
2: takes_reservations
3: outdoor_seating
4: restaurant_is_expensive
5: has_alcohol
6: has_table_service
7: ambience_is_classy
8: good_for_kids 



You can use the [editor on GitHub](https://github.com/akshaym96/Computer-Vision-Project/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](file.path)
```


### Method -1 SIFT, Random Forest


In this method, features for both train and test images were extracted using SIFT. Then these descriptors were used to build clusters using K-Means. The clusters from K-Means are used to build the histograms for the train and test images. The extracted histograms are fed to 9 Random Forest Classifier for training and testing one 9 different available labels.

Feature Extraction time:-
- Train Images:-   15 hours 37 minutes 37 seconds
- Test Images:-     1 hour 22 minutes 26 secs.
- K-means clustering :-  3 hours  7 minutes 48 secs.
- Training time:-  1 minute 34 secs.
- Testing Time:-  30.5867 secs.
Results:- 
Accuracy:-
Strict Match:-  36.7031 %
One Mismatch:-  40.8372%
Two Mismatch:-  51.2016%



For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/akshaym96/Computer-Vision-Project/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
