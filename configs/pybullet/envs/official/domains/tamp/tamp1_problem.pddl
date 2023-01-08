(define (problem tamp-1)
	(:domain workspace)
	(:objects
		rack - unmovable
		cyan_box - box
		yellow_box - box
		red_box - box
	)
	(:init
		(on rack table)
		(on cyan_box table)
		(on yellow_box table)
		(on red_box table)
	)
	(:goal (and
			(on cyan_box rack)
			(on yellow_box rack)
			(on red_box rack)
	))
)
