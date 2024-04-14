import os
import glob

post_dir = '_posts/'
cat_dir = '_categories/'
tag_dir = 'tags/'

def get_cat_tags(post_dir=post_dir):
    filenames = glob.glob(post_dir + "*md")
    all_cats, all_tags = set(), set()
    
    for file in filenames:
        with open(file, 'r', encoding='utf8') as f:
            dash_count = 0
            for line in f:
                curr_line = line.strip()
                if curr_line == '---':
                    dash_count += 1
                    if dash_count == 2:
                        break
                elif curr_line[:8] == 'category':
                    substr = curr_line[8:].strip(': ][').split(',')
                    substr = map(str.strip, substr)
                    all_cats.update(substr)
                elif curr_line[:3] == 'tag':
                    substr = curr_line[3:].strip(': ][').split(',')
                    substr = map(str.strip, substr)
                    all_tags.update(substr)
    return all_cats, all_tags
    
 
def create_cats_tags(all_cats, all_tags, cat_dir=cat_dir, tag_dir=tag_dir ):
    if not os.path.exists(cat_dir):
        os.makedirs(cat_dir)
    
    old_cats = glob.glob(cat_dir + '*.md')
    for cat in old_cats:
        os.remove(cat)
    
    for cat in all_cats:
        cat_filename = cat_dir + cat + '.md'
        with open(cat_filename, 'w') as f:
            contents = '---\ncatname: "' + cat + '"\nlayout: "category"\npermalink: "category/' + cat + '"\n---'
            f.write(contents)
    
    if not os.path.exists(tag_dir):
        os.makedirs(tag_dir)
    
    old_tags = glob.glob(tag_dir + '*.md')
    for tag in old_tags:
        os.remove(tag)
    
    for tag in all_tags:
        tag_filename = tag_dir + tag + '.md'
        with open(tag_filename, 'w') as f:
            contents = '---\ntagname: "' + tag + '"\nlayout: "tagpage"\npermalink: "tag/' + tag + '"\n---'
            f.write(contents)


if __name__ == "__main__":
    all_cats, all_tags = get_cat_tags(post_dir)
    create_cats_tags(all_cats, all_tags, cat_dir, tag_dir)