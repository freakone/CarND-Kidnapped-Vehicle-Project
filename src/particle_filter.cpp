/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <limits>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	num_particles = 100;

	std::default_random_engine generator;
	std::normal_distribution<double> gauss_x(x, std[0]);
	std::normal_distribution<double> gauss_y(y, std[1]);
	std::normal_distribution<double> gauss_t(theta, std[2]);
	
	for (int i = 0; i < num_particles; i++) {

		Particle p;
		p.id = i;
		p.x = gauss_x(generator);
		p.y = gauss_y(generator);
		p.theta = gauss_t(generator);
		p.weight = 1.0;
		
		particles.push_back(p);
		weights.push_back(p.weight);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	std::default_random_engine generator;
	std::normal_distribution<double> gauss_x(0.0, std_pos[0]);
	std::normal_distribution<double> gauss_y(0.0, std_pos[1]);
	std::normal_distribution<double> gauss_t(0.0, std_pos[2]);

	for (auto& p : particles) {

		if(fabs(yaw_rate) < std::numeric_limits<double>::epsilon()) //if yaw is zero
		{
			const double vdt = velocity * delta_t;

			p.x +=  vdt * cos(p.theta) + gauss_x(generator);
			p.y += vdt * sin(p.theta) + gauss_y(generator);
			p.theta += gauss_t(generator);	
		}
		else
		{
	      	const double phi = p.theta + delta_t * yaw_rate;
			const double vy = velocity / yaw_rate;

			p.x += vy * (sin(phi) -  sin(p.theta)) + gauss_x(generator);
			p.y +=  vy * (cos(p.theta) - cos(phi)) + gauss_y(generator);
			p.theta = phi + gauss_t(generator);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (auto& obs : observations) {

		double error = std::numeric_limits<double>::max();

		for (int j = 0; j < predicted.size(); j++){
			const double current_error = dist(predicted[j].x,  predicted[j].y, obs.x, obs.y);

			if (current_error < error) {
				obs.id = j;
				error = current_error;
			}
		}
		
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (int i = 0; i < num_particles; i++) {
		
		vector<LandmarkObs> map_observations;
		for (auto& obs : observations){

			const double tx = particles[i].x + obs.x * cos(particles[i].theta) - obs.y * sin(particles[i].theta);
			const double ty = particles[i].y + obs.y * cos(particles[i].theta) + obs.x * sin(particles[i].theta);

			LandmarkObs observation = { obs.id, tx, ty};
			map_observations.push_back(observation);
		}

		vector<LandmarkObs> landmarks;
		for (auto& map_mark : map_landmarks.landmark_list) {

			const double distance = dist(particles[i].x, particles[i].y, map_mark.x_f, map_mark.y_f);

			if (distance < sensor_range) {
				LandmarkObs landmark_in_range = { map_mark.id_i, map_mark.x_f, map_mark.y_f };
				landmarks.push_back(landmark_in_range);
			}
		}

		dataAssociation(landmarks, map_observations);

		particles[i].weight = 1.0;

		for (auto& obs : map_observations) {

			const double a = 0.5 / (pow(std_landmark[0], 2)) * pow(obs.x - landmarks[obs.id].x, 2);
			const double b = 0.5 / (pow(std_landmark[1], 2)) * pow(obs.y - landmarks[obs.id].y, 2);

			particles[i].weight *= exp(-(a + b)) / sqrt( 2.0 * M_PI * std_landmark[0] * std_landmark[1]);
		}

		weights[i] = particles[i].weight;		
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::default_random_engine generator;
	std::discrete_distribution<int> ind(this->weights.begin(), this->weights.end());

	vector<Particle> resampled;

	for (int i = 0; i < num_particles; i++) {

		const int idx = ind(generator);

		Particle p;
		p.id = idx;
		p.x = particles[idx].x;
		p.y = particles[idx].y;
		p.theta = particles[idx].theta;
		p.weight = 1.0;

		resampled.push_back(p);
	}

	particles.swap(resampled);
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
