__host__ __device__ float square(float num) {return num*num;}

__device__ void wray(int np, int nr, int *line, float *weight, int *numb){
		float sorx, sory, cf, sf, rcosphi, tanphi, rtnphi, c, fp, w, yc;
		int ix, iy, incx, incy, index, next, flip, flipxy;
		/* get the info of the line (np,nr) */
		float sinth = sin(PI/NPROJ*np);
		float costh = cos(PI/NPROJ*np);

		int rwidth = 1; /* space between detectors */
		float tmp_length = (float)((nr-MIDRAY)*rwidth);
		sorx = -sinth*tmp_length;
		sory = costh*tmp_length;
		cf = costh;
		sf = sinth;


		if (sf < -F_TOL) {
			sf = -sf;
			cf = -cf;
		}

		flip = 0;
		if (sf * cf < -F_TOL*F_TOL) {
			flip = 1;
			cf = -cf;
		}

		if (flip > 0) {
			sorx = -sorx;
		}

		if (fabs(sf) <= F_TOL) {
			if (sf < 0) sf = -F_TOL*F_TOL;
			else sf = F_TOL*F_TOL;
		}

		if (fabs(cf) <= F_TOL) {
			if (cf < 0) cf = -F_TOL*F_TOL;
			else cf = F_TOL*F_TOL;
		}

		if (fabs(cf) > F_TOL) tanphi = sf / cf;
		else tanphi = 1/(F_TOL*F_TOL);

		if (fabs(sf) > F_TOL) rtnphi = cf / sf;
		else rtnphi = 1/(F_TOL*F_TOL);

		flipxy = 0;
		if ( tanphi > 1+F_TOL ) {
			flipxy = 1;
			tanphi = 1/tanphi;
			rtnphi = 1/rtnphi;
			sf = sory;
			sory = sorx;
			sorx = sf;
		}

		rcosphi = ( sqrt( 1 + tanphi* tanphi));

		sory += IMGSIZE / 2.0;
		sorx += IMGSIZE / 2.0;
		c = sory - sorx * tanphi;

		if ((c - IMGSIZE) >= GRID_TOL ) return;
		if (c < -GRID_TOL) {
			c = -c * rtnphi;
			if((c - IMGSIZE) >= GRID_TOL) return;
			ix = (int) c;
			iy = 0;
			fp = ((float) (ix + 1)) - c;
			yc = fp * tanphi;
			yc += -1.0;
		}
		else {
			ix = 0;
			iy = (int) c;
			fp = ((float) (iy + 1)) - c;
			yc = -fp;
			yc += tanphi;
		}

		if (yc>=0) next = 1;
		else next = 2;

		/* calculate the initial index and the increment of index */
		if (flipxy > 0){
			index = iy + ix * IMGSIZE;
			incx = IMGSIZE;
			incy = 1;
			if (flip>0){
				index = IMGSIZE - 1 - iy + ix * IMGSIZE ;
				incx = IMGSIZE;
				incy = -1;
			}
		}
		else {
			index = iy * IMGSIZE + ix;
			incx = 1;
			incy = IMGSIZE;
			if (flip>0) {
				index = IMGSIZE - 1 - ix + iy * IMGSIZE ;
				incx = -1;
				incy = IMGSIZE;
			}
		}

		for(;;) {
			switch(next) {
			case 1:
				if (iy+1>= IMGSIZE || ix<0) return;

				w = rcosphi * (1 - yc * rtnphi );
				weight[*numb] = w;
				line[*numb] = index;
				++(*numb);
				iy ++;
				index += incy;

				if (ix+1>=IMGSIZE) return;

				w = rcosphi * ( yc * rtnphi );
				weight[*numb] = w;
				line[*numb] = index;
				++(*numb);
				ix ++;
				index += incx;

				yc += tanphi - 1.0;
				if (yc>=0) next = 1;
				else next = 2;
				break;
			case 2:
				if (iy>= IMGSIZE || ix+1>=IMGSIZE || ix<0) return;

				w = rcosphi;
				weight[*numb] = w;
				line[*numb] = index;
				++(*numb);
				ix ++;
				index += incx;

				yc += tanphi;
				if (yc>=0) next = 1;
				else next = 2;
				break;
			}
		}
	
}

__global__ void XCT_Reconstruction(float *f, float *v, float *g, int *angle, int *position, float lambda, 
	Node *records, Lock lock, Global_counter counter){

	int i;
	i=blockIdx.x*blockDim.x+threadIdx.x;
	int np=angle[i];
	int nr=position[i];

	int numb = 0;
	int line[2*IMGSIZE];
	float weight[2*IMGSIZE];
	float d[2*IMGSIZE];
	float cached_f[2*IMGSIZE];
	int parent_id;

	lock.lock();
	parent_id = *(counter.counter);
	wray(np, nr, line, weight, &numb);

	float Af = 0.0f;
	for (i = 0; i<numb; ++i) {
		int x = line[i]/IMGSIZE+1;
		int y = line[i]%IMGSIZE+1;
		int ind=x*(IMGSIZE+2)+y;
		cached_f[i] = f[ind];
		Af += cached_f[i]*weight[i];
	}
	lock.unlock();

	Af -= g[np*NRAY+nr];

	for (i = 0; i<numb; ++i)
		d[i] = -Af*weight[i];
	

	for (i = 0; i<numb; ++i) {
		int x = line[i]/IMGSIZE+1;
		int y = line[i]%IMGSIZE+1;
		int ind=x*(IMGSIZE+2)+y;

		float tmp = cached_f[i] + lambda*d[i];
		if (tmp<0) tmp = 0;
		if (tmp>255) tmp = 255;
		lock.lock();
		if(*(counter.counter) < MAX_RECORDS)
		{
			records[*(counter.counter)].pre = parent_id;
			records[*(counter.counter)].pos = ind;
			records[*(counter.counter)].delta = lambda * d[i];
			*(counter.counter) += 1;
		}
		lock.unlock();
	}

}
