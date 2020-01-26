/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"
#include <math.h>
#include <algorithm>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::discrete_distribution;
using std::uniform_int_distribution;

std::default_random_engine generator;

const double YAW_RATE_THRESHOLD = 0.00001;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  if (!this->is_initialized) {
      num_particles = 100;  // number of particles

      normal_distribution<double> dis_x(x,std[0]);
      normal_distribution<double> dis_y(y,std[1]);
      normal_distribution<double> dis_theta(theta,std[2]);

      for (auto i = 0; i < num_particles; ++i) {
          Particle pr;
          pr.id = i;
          pr.x = dis_x(generator);
          pr.y = dis_y(generator);
          pr.theta = dis_theta(generator);
          pr.weight = 1;
          particles.emplace_back(pr);
      }

      this->is_initialized = true;
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
    normal_distribution<double> dis_x(0,std_pos[0]);
    normal_distribution<double> dis_y(0,std_pos[1]);
    normal_distribution<double> dis_theta(0,std_pos[2]);

    for (auto& pr : particles) {
        if(abs(yaw_rate) > YAW_RATE_THRESHOLD) {
            pr.x += velocity / yaw_rate * (sin(pr.theta + yaw_rate * delta_t) - sin(pr.theta)) + dis_x(generator);
            pr.y += velocity / yaw_rate * (cos(pr.theta) - cos(pr.theta + yaw_rate * delta_t)) + dis_y(generator);
            pr.theta += yaw_rate * delta_t + dis_theta(generator);
        }
        else {
            pr.x += velocity * delta_t * cos(pr.theta) + dis_x(generator);
            pr.y += velocity * delta_t * sin(pr.theta) + dis_y(generator);
            pr.theta += dis_theta(generator);
        }
    }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (auto& observ : observations) {
      double min_dist = 100000000;
      int pred_id = 0;
      for (const auto& pred : predicted) {
         auto dx = pred.x - observ.x;
         auto dy = pred.y - observ.y;
         auto dist = dx * dx + dy * dy;
         pred_id = dist < min_dist? pred.id : pred_id;
         min_dist = std::min(min_dist, dist);
      }
      observ.id = pred_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
    for (auto & particle : particles) {

        vector<LandmarkObs> observations_in_map_coords;
        for (const auto& observ : observations) {
            double x_map = particle.x + cos(particle.theta) * observ.x - sin(particle.theta) * observ.y;
            double y_map = particle.y + sin(particle.theta) * observ.x + cos(particle.theta) * observ.y;
            observations_in_map_coords.emplace_back(LandmarkObs{observ.id, x_map, y_map});
        }

        vector<LandmarkObs> predicted;
        for (const auto& landmark : map_landmarks.landmark_list) {
            if ((fabs(particle.x - landmark.x_f) <= sensor_range) &&
                (fabs(particle.y - landmark.y_f) <= sensor_range)) {
                predicted.emplace_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
            }
        }

        dataAssociation(predicted, observations_in_map_coords);

        particle.weight = 1.0;
        for (const auto& observ : observations_in_map_coords) {
            if(observ.id > -1) {
                LandmarkObs landmark = {};
                for (const auto& pred : predicted) {
                    if (pred.id == observ.id) {landmark = pred; break;}
                }
                // calculate normalization term
                double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
                // calculate exponent
                double exponent = (pow(observ.x - landmark.x, 2) / (2 * pow(std_landmark[0], 2)))
                                + (pow(observ.y - landmark.y, 2) / (2 * pow(std_landmark[1], 2)));
                // calculate weight using normalization terms and exponent
                particle.weight *= gauss_norm * exp(-exponent);
            }
            else {
                particle.weight *= std::numeric_limits<double>::min();
            }
        }
    }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
    vector<Particle> resampled_particles;

    weights.clear();

    for(const auto& particle : particles){weights.emplace_back(particle.weight);}

    discrete_distribution<> part_dist_idx(weights.begin(), weights.end());

    for(auto i = 0; i < num_particles; ++i) {
        auto idx = part_dist_idx(generator);
        resampled_particles.emplace_back(particles[idx]);
    }
    particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}