#pragma once 

class MathsUtils {
	public:
		static inline float _fmin(float a, float b) {
			return (a < b) ? a : b;
		}
		
		static inline float _fmax(float a, float b) {
			return (a > b) ? a : b;
		}

		static inline float _fabs(float a) {
			return (a < 0) ? -a : a;
		}

		static inline double _min(double a, double b) {
			return (a < b) ? a : b;
		}

		static inline double _max(double a, double b) {
			return (a > b) ? a : b;
		}

		template<typename T>
			static inline void _swap(T& a, T& b) {
				T tmp = a;
				a = b;
				b = tmp;
			}

		static inline double _abs(double a) {
			return (a < 0) ? -a : a;
		}
};
