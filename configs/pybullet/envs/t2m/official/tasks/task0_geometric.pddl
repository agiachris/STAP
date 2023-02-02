(define (problem lh_0)
	(:domain geometric_workspace)
	(:objects
        rack - receptacle
		blue_box - box
        cyan_box - box 
        yellow_box - box
	)
	(:init
		(on rack table)
		(on blue_box table)
        (on cyan_box table)
        (on yellow_box table)
        ; Geometric facts
        (inworkspace table)
        (inworkspace rack)
        (inworkspace blue_box)
        (inworkspace cyan_box)
        (inworkspace yellow_box)        
	)
	(:goal (and
        (on blue_box rack)
        (on cyan_box rack)
        (on yellow_box rack)
	))
)
