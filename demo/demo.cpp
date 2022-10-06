/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>

// DBoW2
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

using namespace DBoW2;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void loadFeatures(vector<vector<std::pair<cv::Mat, int>>> &features);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void testDatabase(const vector<vector<std::pair<cv::Mat, int>>> &features, SemanticOrbVocabulary &voc);


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// number of training images
const int NIMAGES = 6;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// ----------------------------------------------------------------------------

int main()
{
  vector<vector<std::pair<cv::Mat, int>>> features;
  loadFeatures(features);

  // Load the vocabulary from vocab text file
  SemanticOrbVocabulary voc;
  voc.loadFromTextFile("/home/nate/Development/Semantic-DBoW2/vocabulary/ORBvoc.txt");

  testDatabase(features, voc);

  return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(vector<vector<std::pair<cv::Mat,int>>> &features)
{
  features.clear();
  features.reserve(NIMAGES);

  cv::Ptr<cv::ORB> orb = cv::ORB::create();

  cout << "Extracting ORB features..." << endl;
  for(int i = 0; i < NIMAGES; ++i)
  {
    stringstream ss;
    ss << "/home/nate/Development/Semantic-DBoW2/demo/images/image" << i << ".png";

    cv::Mat image = cv::imread(ss.str(), 0);
    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb->detectAndCompute(image, mask, keypoints, descriptors);

    std::vector<std::pair<cv::Mat,int>> imgDescriptors;

    for( int j = 0; j < descriptors.rows; j++ )
    {
        imgDescriptors.emplace_back(std::make_pair(descriptors.row(j), -1));
    }

    features[i] = imgDescriptors;
  }
}

// ----------------------------------------------------------------------------

void testVocCreation(const vector<vector<cv::Mat > > &features)
{
  // branching factor and depth levels
  const int k = 9;
  const int L = 3;
  const WeightingType weight = TF_IDF;
  const ScoringType scoring = L1_NORM;

  OrbVocabulary voc(k, L, weight, scoring);

  cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
  voc.create(features);
  cout << "... done!" << endl;

  cout << "Vocabulary information: " << endl
  << voc << endl << endl;

  // lets do something with this vocabulary
  cout << "Matching images against themselves (0 low, 1 high): " << endl;
  BowVector v1, v2;
  for(int i = 0; i < NIMAGES; i++)
  {
    voc.transform(features[i], v1);
    for(int j = 0; j < NIMAGES; j++)
    {
      voc.transform(features[j], v2);

      double score = voc.score(v1, v2);
      cout << "Image " << i << " vs Image " << j << ": " << score << endl;
    }
  }

  // save the vocabulary to disk
  cout << endl << "Saving vocabulary..." << endl;
  voc.save("small_voc.yml.gz");
  cout << "Done" << endl;
}

// ----------------------------------------------------------------------------

void testDatabase(const std::vector<std::vector<std::pair<cv::Mat, int>>> &features, SemanticOrbVocabulary &voc)
{
  cout << "Creating a small database..." << endl;

  std::string classFile = "/home/nate/Development/Semantic-DBoW2/demo/config/labels_test.json";
  SemanticOrbDatabase db(voc, classFile, false, 0); // false = do not use direct index

  // TODO Nate: add semantic ORB features to the database
  for(int i = 0; i < NIMAGES; i++)
  {
    db.add(features[i]);
  }

  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;

  // and query the database
  cout << "Querying the database: " << endl;

  QueryResults ret;
  for(int i = 0; i < NIMAGES; i++)
  {
    db.query(features[i], ret, 6);

    // ret[0] is always the same image in this case, because we added it to the
    // database. ret[1] is the second best match.

    cout << "Searching for Image " << i << ". " << ret << endl;
  }

  cout << endl;
}

// ----------------------------------------------------------------------------


