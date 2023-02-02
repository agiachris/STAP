(define (problem lh_lg_0)
	(:domain geometric_workspace)
	(:objects
        rack - receptacle
        red_box - box
		blue_box - box
        yellow_box - box
        cyan_box - box 
	)
	(:init
		(on rack table)
        (on red_box table)
		(on blue_box table)
        (on yellow_box table)
        (on cyan_box table)
        ; Geometric facts
        (inworkspace table)
        (inworkspace rack)
        (inworkspace red_box)
        (inworkspace blue_box)
        (inworkspace yellow_box)        
        (inworkspace cyan_box)
	)
	(:goal (or
        (and
            (on red_box rack)
            (on blue_box rack)
            (on yellow_box rack)
            (on cyan_box table)
        )
        (and
            (on red_box rack)
            (on blue_box rack)
            (on yellow_box table)
            (on cyan_box rack)
        )
        (and
            (on red_box rack)
            (on blue_box table)
            (on yellow_box rack)
            (on cyan_box rack)
        )
        (and
            (on red_box table)
            (on blue_box rack)
            (on yellow_box rack)
            (on cyan_box rack)
        )
    ))
)
