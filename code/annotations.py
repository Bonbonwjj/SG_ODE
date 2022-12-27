import numpy as np
import pickle
import argparse


parser = argparse.ArgumentParser()
#path set
parser.add_argument('--frames_path', default='/data/scene_understanding/action_genome/frames/')
parser.add_argument('--annotations_path', default='/data/scene_understanding/action_genome/annotations/')

parser.add_argument('--nonperson_filer',default=True)


def build_vocab(annotations_path):

    vocab={}

    object_classes_itn={}#idex to name
    object_classes_itn[0]='__image__'
    object_classes_nti={}
    object_classes_nti['__image__']=0
    with open(annotations_path + 'object_classes.txt', 'r') as f:
        i=1
        for line in f.readlines():
            line=line.strip('\n')
            object_classes_itn[i]=line
            object_classes_nti[line]=i
            i+=1

    object_classes_itn[9] = 'closet/cabinet'
    object_classes_itn[11] = 'cup/glass/bottle'
    object_classes_itn[23] = 'paper/notebook'
    object_classes_itn[24] = 'phone/camera'
    object_classes_itn[31] = 'sofa/couch'

    del object_classes_nti['closetcabinet'],object_classes_nti['cupglassbottle'],object_classes_nti['papernotebook'],
    del object_classes_nti['phonecamera'],object_classes_nti['sofacouch']

    object_classes_nti['closet/cabinet']=9
    object_classes_nti['cup/glass/bottle']=11
    object_classes_nti['paper/notebook']=23
    object_classes_nti['phone/camera']=24
    object_classes_nti['sofa/couch']=31


    relationship_classes_itn = {}
    relationship_classes_itn[0]='__in_image__'
    relationship_classes_nti={}
    relationship_classes_nti['__in_image__']=0

    with open(annotations_path+'relationship_classes.txt', 'r') as f:
        i=0
        j=1
        for line in f.readlines():
            line = line.strip('\n')
            if i>=3 and i<9:
                relationship_classes_itn[j]=line
                relationship_classes_nti[line]=j
                j+=1
            i+=1
    vocab['object_classes_itn']=object_classes_itn
    vocab['object_classes_nti']=object_classes_nti
    vocab['relationship_classes_itn']=relationship_classes_itn
    vocab['relationship_classes_nti']=relationship_classes_nti

    print('make vocab successfully')
    return vocab


def get_info(annotations_path,nonperson_filter=True,vocab=None):

    kf_info={}

    with open(annotations_path+'person_bbox.pkl', 'rb') as f:
        person_bbox = pickle.load(f)


    with open(annotations_path+'object_bbox_and_relationship.pkl','rb') as f:
        object_bbox_and_relationship=pickle.load(f)

    frame_list=[]
    with open(annotations_path + 'frame_list.txt', 'r') as f:
        for frame in f:
            frame_list.append(frame.rstrip('\n'))

    count = 0
    for id in frame_list:
            if nonperson_filter:
                if person_bbox[id]['bbox'].shape[0] == 0:
                    count += 1
                    continue
            else:
                kf_info[id]={'bbox':[],'obj':[0],'relationship':[]}
            kf_info[id] = {'bbox': [person_bbox[id]['bbox'][0]], 'obj': [1],'relationship':[]}#add person_bbox and person as object 1

    print('Delete %d nonperson_frames' % count)

    one_obj_list=[]
    for id in kf_info:
        count = 0
        for object in object_bbox_and_relationship[id]:
            if object['visible']==True:
                kf_info[id]['obj'].append(vocab['object_classes_nti'][object['class']])#add object according to vocab
                count+=1
                # bbox_mode 'xywh'→'xyxy'
                kf_info[id]['bbox'].append(np.array([object['bbox'][0], object['bbox'][1],
                                                     object['bbox'][0] + object['bbox'][2], object['bbox'][1] + object['bbox'][3]]))

                # add relationship: each object has a relationship spatial no relationship then set 0 __in_image__
                if object['spatial_relationship']:
                    kf_info[id]['relationship'].append(
                        vocab['relationship_classes_nti'][object['spatial_relationship'][0].replace('_','')])

                else:
                    kf_info[id]['relationship'].append(0)

            else:
                continue

        if count==0:
            one_obj_list.append(id)

    for id in one_obj_list:
        kf_info.pop(id)

    print('Delete %d one object Key Frame'%len(one_obj_list))
    print('Key frame info successfully')


    video_info = {}

    length=0
    max_length=0
    for id in kf_info:
        video_name = id.split('/')[0]
        kf_info[id]['id'] = id.split('/')[1]
        length=len(kf_info[id]['relationship'])

        if length>max_length:
            max_length=length
        if video_name not in video_info.keys():
            video_info[video_name] = [kf_info[id]]
        else:
            video_info[video_name].append(kf_info[id])

    few_frame_video_list=[]
    for k,v in video_info.items():
        num=len(v)
        if num<=4:
            few_frame_video_list.append(k)

    for name in few_frame_video_list:
        video_info.pop(name)

    few_frame_list=[]
    for id in kf_info:
        if id.split('/')[0] in few_frame_video_list:
            few_frame_list.append(id)

    for id in few_frame_list:
        kf_info.pop(id)

    print('Delete %d Video with few Frames'% len(few_frame_video_list))

    print('make video info successfully')

    #make train set
    train=[]
    train_list=[]
    for k,v in object_bbox_and_relationship.items():
        if v[0]['metadata']['set']=='train':
            train.append(k.split('/')[0])

    for t in train:
        if t in video_info.keys():
            if t not in train_list:
                train_list.append(t)

    print('make train list succesffully')

    test=[]
    test_list = []
    for k, v in object_bbox_and_relationship.items():
        if v[0]['metadata']['set'] == 'test':
            test.append(k.split('/')[0])

    for t in test:
        if t in video_info.keys():
            if t not in test_list:
                test_list.append(t)

    print('make test list succesffully')

    max_length=2*max_length+1

    return kf_info,video_info,train_list,test_list,max_length



if __name__ == '__main__':
    args = parser.parse_args()
    vocab=build_vocab(args.annotations_path)
    kf_info,video_info,train_list,test_list,max_length=get_info(args.annotations_path, args.nonperson_filer,vocab)










