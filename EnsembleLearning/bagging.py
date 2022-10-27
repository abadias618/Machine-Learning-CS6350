"""
@author: abdias baldiviezo
"""
import pandas as pd
from math import log2
from random import randrange
from matplotlib import pyplot as plt

class Decision_Tree():
    """
    data: pandas DF made by the load_data() function.
    depth: int < NumberOfColumns Default: number of columns - 1\n
    ig_type: entropy, me, gi Default: entropy.
    """
    def __init__(self, data, depth = None, ig_type = 'entropy'):
        self.data = data
        self.depth = depth or len(data.columns)-1
        self.ig_type = ig_type
        
        root = Node()
        root.makeRoot()
        
        self.tree_structure, _ = id3(self.data,
                                     self.data.columns.tolist(),
                                     root,
                                     self.depth,
                                     self.ig_type)
    
    
    def predict(self, data):
        """
        data: pandas DF made by the load_data() function
        """
        predictions = []
        for i in range(0, len(data)):
            
            n = self.tree_structure
            # Root node
            while not n.isLeaf:
                if n.isRoot:
                    if self.depth == 1:
                        n = n.children[0]
                        break
                    print('i',i,'n.attr',n.attribute)
                    decision = data.loc[i, n.attribute]
                elif n.children[0].attribute == None:
                    n = n.children[0]
                    break
                else:
                    #print('node attr',n.children[0].attribute)
                    decision = data.loc[i, n.children[0].attribute]
                    #print('node decision', decision)
                
                found = False    
                for child in n.children:
                    if child.decision == decision:
                        #take route
                        n = child
                        found = True
                        break
                if not found:
                    # if path for that combination doesn't exist due to low
                    # depth or some other reason, introduce the most common
                    # label as the prediction
                    most_common_label = data.iloc[:,len(data.columns)-1].value_counts().index.tolist()
                    decision = most_common_label[0]
                    break
            # out of loop with n being a leaf assign decision to prediction
            predictions.append(n.decision)
        return predictions
    
    def recursive_print(self, tree_structure):
        if tree_structure.isRoot:    
            print('attribute',tree_structure.attribute,
                  'self',tree_structure)
        elif tree_structure.isLeaf:
            print('\t\tdecision',tree_structure.decision,
                  'attribute',tree_structure.attribute,
                  'self',tree_structure,
                  'parent',tree_structure.parent,
                  'leaf',tree_structure.isLeaf)
        else:
            print('\tdecision',tree_structure.decision,
                  'attribute',tree_structure.attribute,
                  'self',tree_structure,
                  'parent',tree_structure.parent)
        for e in tree_structure.children:
            print('child#',tree_structure.children.index(e))
            self.recursive_print(e)
            
    def print_tree(self):
        self.recursive_print(self.tree_structure)
        
class Node:
    def __init__(self):
        self.children = []
        self.parent = None
        self.attribute = None
        self.decision = None
        self.isRoot = False
        self.isLeaf = False
        
    def setParent(self, parent):
        self.parent = parent
        
    def setDecision(self, decision_name):
        self.decision = decision_name
        
    def setAttribute(self, attribute_name):
        self.attribute = attribute_name
        
    def add_child(self, child):
        self.children.append(child)
        
    def makeRoot(self):
        self.isRoot = True
    
    def makeLeaf(self):
        self.isLeaf = True
        
class Leaf:
    def __init__(self, parent, decision):
        self.parent = parent
        self.decision = decision

def id3(S, available_columns, node, depth, ig_type):
    # print('-------------------')
    # print('ac', available_columns)
    # print('depth',depth)
    # print('len S', len(S))
    # print(S)
    most_common_element = S.iloc[:,len(S.columns)-1].value_counts().index.tolist()
    
    if (depth <= 1
        or len(most_common_element) <= 1 
        or len(available_columns) <= 1):
        sub_node = Node()
        sub_node.setParent(node)
        sub_node.setDecision(most_common_element[0])
        node.add_child(sub_node)
        sub_node.makeLeaf()
        return node, available_columns
    if len(S) < 1:
        return node, available_columns
    
    ig = []
    if ig_type == 'entropy':
        ig = entropy_info_gain(S, available_columns)
    elif ig_type == 'me':
        ig = me(S, available_columns)
    elif ig_type == 'gi':
        ig = gi(S, available_columns)
    # split on
    split_column_index = ig.index(max(ig))
    # print('i', split_column_index, 'name', S.columns[split_column_index])
    split_column = S.columns[split_column_index]
    # pop split column from available columns
    available_columns.pop(available_columns.index(S.columns[split_column_index]))
    # root Node
    if node.isRoot:
        node.setAttribute(split_column)
    # split
    attribute_counts = S.loc[:, split_column].value_counts()
    count = 1
    for attr, _ in attribute_counts.iteritems():
        # Make Subset(Sn) with all columns except the one that is being split on
        if len(S) < 1:
            continue
        # make Node
        sub_node = Node()
        sub_node.setParent(node)
        sub_node.setAttribute(split_column)
        sub_node.setDecision(attr)
        node.add_child(sub_node)
        # split S
        Sn = S.loc[ S.loc[:, split_column ] == attr, S.columns != split_column ]
        
        count += 1
        
        _, available_columns = id3(Sn, available_columns, sub_node, depth-1, ig_type)
        
    return node, available_columns

def entropy_info_gain(S, available_columns):
    # Label Index = len(S.columns)-1
    general_counts = S.iloc[:,len(S.columns)-1].value_counts()
    # IG of the whole set (S).
    H = 0
    for e in general_counts:
        H -= ((e/len(S))*log2(e/len(S)))
    
    ig = []
    # FOR all columns except last
    for i in range(len(S.columns)-1):
        #if column is not available introduce a 0
        if S.columns[i] not in available_columns:
            ig.append(-100)
            continue
        attribute_counts = S.iloc[:,i].value_counts()
        expected = 0
        # FOR each attribute in column counts
        for attr, _ in attribute_counts.iteritems():
            # Make Subset(Sn) with filtered by attribute column and label column
            Sn = S.loc[ S.iloc[:,i] == attr, S.columns[len(S.columns)-1] ]
            counts = Sn.value_counts()
            Hn = 0
            # FOR each element in Sn counts
            for e in counts:
                Hn -= ((e/len(Sn))*log2(e/len(Sn)))
            # expected Entropy
            expected += (len(Sn)/len(S))*Hn
         
        ig.append(H - expected)
    # IG 
    return ig

def me(S, available_columns):
    # Label Index = len(S.columns)-1
    general_counts = S.iloc[:,len(S.columns)-1].value_counts()
    # IG of the whole set (S).
    ME = 0
    minimum_ME = []
    for e in general_counts:
        if (e/len(S)) == 1.0:
            minimum_ME.append(0)
        else:
            minimum_ME.append((e/len(S)))
    ME = min(minimum_ME)
    ig = []
    # FOR all columns except last
    for i in range(len(S.columns)-1):
        #if column is not available introduce a 0
        if S.columns[i] not in available_columns:
            ig.append(-100)
            continue
        attribute_counts = S.iloc[:,i].value_counts()
        expected = 0
        # FOR each attribute in column counts
        for attr, _ in attribute_counts.iteritems():
            # Make Subset(Sn) with filtered by attribute column and label column
            Sn = S.loc[ S.iloc[:,i] == attr, S.columns[len(S.columns)-1] ]
            counts = Sn.value_counts()
            MEn = 0
            minimum_MEn = []
            # FOR each element in Sn counts
            for e in counts:
                if (e/len(Sn)) == 1.0:
                    minimum_MEn.append(0)
                else:
                    minimum_MEn.append((e/len(Sn)))
            MEn = min(minimum_MEn)
            # expected
            expected += (len(Sn)/len(S))*MEn
        ig.append(ME - expected)
    # IG
    return ig

def gi(S, available_columns):
    
    # Label Index = len(S.columns)-1
    general_counts = S.iloc[:,len(S.columns)-1].value_counts()
    # IG of the whole set (S).
    GI = 1
    for e in general_counts:
        GI -= (e/len(S))**2
    
    ig = []
    # FOR all columns except last
    for i in range(len(S.columns)-1):
        #if column is not available introduce a 0
        if S.columns[i] not in available_columns:
            ig.append(-100)
            continue
        attribute_counts = S.iloc[:,i].value_counts()
        expected = 0
        # FOR each attribute in column counts
        for attr, _ in attribute_counts.iteritems():
            # Make Subset(Sn) with filtered by attribute column and label column
            Sn = S.loc[ S.iloc[:,i] == attr, S.columns[len(S.columns)-1] ]
            counts = Sn.value_counts()
            GIn = 1
            # FOR each element in Sn counts
            for e in counts:
                GIn -= (e/len(Sn))**2
            # expected Entropy
            expected += (len(Sn)/len(S))*GIn
         
        ig.append(GI - expected)
    # IG 
    return ig
def sample_data_w_replacement(S):
    # samples = S.sample(n=len(S), replace=True)
    # samples.reset_index(drop=True)
    #print('sample\n',sample)
    return S

def baggedTrees(T, S):
    trees = []
    for i in range(0,T):
        #Sn = sample_data_w_replacement(S)
        tree = Decision_Tree(S,ig_type='entropy')
        trees.append(tree)
    return trees

def predict_w_baggedTrees(trees, S):
    trees_predictions = []
    predictions = []
    for i in range(0,len(S)):
        for e in trees:
            trees_predictions.append(1*int(e.predict(S.iloc[i])[0]))
        most_common_element = pd.DataFrame(trees_predictions).value_counts().index.tolist()[0]  
        predictions.append(most_common_element)
    
    return predictions

def average_prediction_error(predictions, labels):
    if len(predictions) != len(labels):
        raise RuntimeError('Different predictions & labels sizes',
                           len(predictions),
                           len(labels))
    correct = 0
    for (a,b) in zip(predictions, labels):
        if a == b:
            correct += 1
    return (len(predictions)-correct)/len(predictions)
    
def load_data(filename, replace_numeric=False, replace_unk=False, make_binary_labels=False):
    df = pd.read_csv(filename,header=None)
    df.columns = ['column#'+str(col+1) for col in df.columns]
    if replace_numeric:
        is_numeric_df = df.apply(lambda col: pd.to_numeric(col, errors='coerce').notnull().all())
        # replace numerical with overmedian, undermedian values
        for i in range(len(is_numeric_df)):
            if is_numeric_df[i]:
                median = df.iloc[:,i].median()
                df.iloc[df.iloc[:,i] >= median, i] = 'overmedian'
                df.iloc[df.iloc[:,i] != 'overmedian', i] = 'undermedian'
        #print(df)        
    if replace_unk:
        for i in range(len(df.columns)-1):
            most_common_values_list = df.iloc[:,i].value_counts().index.tolist()
            if most_common_values_list[0] == 'unknown':
                most_common_value = most_common_values_list[1]
            else:
                most_common_value = most_common_values_list[0]
            df.iloc[df.iloc[:,i] == 'unknown', i] = most_common_value
        #print(df)
    if make_binary_labels:
            df.iloc[df.iloc[:,len(df.columns)-1] == 'yes', i] = 1
            df.iloc[df.iloc[:,len(df.columns)-1] != 1, i] = -1
    return df

def main():
            
    #BANK dataset with numerical replacement
    training = load_data('./bank-2/train.csv', replace_numeric=True, make_binary_labels=True)
    #test = load_data('./bank/test.csv', replace_numeric=True)
    errors= []
    x_label = []
    for i in range(1,5):
        x_label.append(i)
        trees = baggedTrees(i, training)
        errors.append(
            average_prediction_error(predict_w_baggedTrees(trees,training),
            training.iloc[:,len(training.columns)-1].tolist())
            )
        
    plt.plot(errors)
    plt.savefig('baggedTreesTest.png')
    
if __name__ == '__main__':
    main()

