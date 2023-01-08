(define (problem tamp-0)
	(:domain workspace)
	(:objects
		rack - unmovable
		hook - tool
		yellow_box - box
		red_box - box
	)
	(:init
		(on rack table)
		(on yellow_box table)
		(on red_box table)
	)
	(:goal (and
			(on yellow_box rack)
			(on red_box rack)
	))
)
