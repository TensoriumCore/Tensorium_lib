#pragma once

#ifdef MORPHEUS_USE_MPI
# include <mpi.h>
#endif

namespace morpheus {
	namespace mpi {

#ifdef MORPHEUS_USE_MPI

		inline void init(int *argc, char ***argv) {
			MPI_Init(argc, argv);
		}

		inline void finalize() {
			MPI_Finalize();
		}

		inline int rank() {
			int rank;
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);
			return rank;
		}

		inline int size() {
			int size;
			MPI_Comm_size(MPI_COMM_WORLD, &size);
			return size;
		}

		inline void barrier() {
			MPI_Barrier(MPI_COMM_WORLD);
		}

		inline void send(const void* data, int count, MPI_Datatype type, int dest, int tag) {
			MPI_Send(data, count, type, dest, tag, MPI_COMM_WORLD);
		}

		inline void recv(void* data, int count, MPI_Datatype type, int source, int tag) {
			MPI_Recv(data, count, type, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		inline void bcast(void* data, int count, MPI_Datatype type, int root) {
			MPI_Bcast(data, count, type, root, MPI_COMM_WORLD);
		}

		inline void allreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype type, MPI_Op op) {
			MPI_Allreduce(sendbuf, recvbuf, count, type, op, MPI_COMM_WORLD);
		}

		inline void reduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype type, MPI_Op op, int root) {
			MPI_Reduce(sendbuf, recvbuf, count, type, op, root, MPI_COMM_WORLD);
		}

		inline void gather(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
				void* recvbuf, int recvcount, MPI_Datatype recvtype, int root) {
			MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, MPI_COMM_WORLD);
		}

		inline void scatter(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
				void* recvbuf, int recvcount, MPI_Datatype recvtype, int root) {
			MPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, MPI_COMM_WORLD);
		}

#else // no MPI


		inline void init(int*, char***) {}
		inline void finalize() {}
		inline int rank() { return 0; }
		inline int size() { return 1; }
		inline void barrier() {}

		template<typename... Args>
			inline void send(Args...) {}

		template<typename... Args>
			inline void recv(Args...) {}

		template<typename... Args>
			inline void bcast(Args...) {}

		template<typename... Args>
			inline void allreduce(Args...) {}

		template<typename... Args>
			inline void reduce(Args...) {}

		template<typename... Args>
			inline void gather(Args...) {}

		template<typename... Args>
			inline void scatter(Args...) {}

#endif // MORPHEUS_USE_MPI

	}
} 
