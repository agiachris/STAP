(define (problem constrained-packing-0)
	(:domain workspace)
	(:objects
		rack - unmovable
		red_box - box
		yellow_box - box
		cyan_box - box
	)
	(:init
		(on rack table)
		(on red_box table)
		(on yellow_box table)
		(on cyan_box rack)
	)
	(:goal (and
		(on red_box rack)
		(on yellow_box rack)
		(on cyan_box rack)
	))
)
