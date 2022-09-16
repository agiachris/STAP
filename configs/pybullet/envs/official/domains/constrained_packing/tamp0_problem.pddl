(define (problem constrained-packing-0)
	(:domain workspace)
	(:objects
		rack - unmovable
		red_box - box
		yellow_box - box
		cyan_box - box
		blue_box - box
	)
	(:init
		(on rack table)
		(on red_box table)
		(on yellow_box table)
		(on cyan_box yellow_box)
		(on blue_box table)
	)
	(:goal (or
		(and
			(on red_box rack)
		)
		(and
			(on cyan_box rack)
		)
		(and
			(on yellow_box rack)
		)
		(and
			(on blue_box rack)
		)
	))
)
