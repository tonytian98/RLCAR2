import cv2


class ImageProcessor:
    def __init__(
        self,
        image_path: str,
        resize=[800, 600],
    ) -> None:
        self.image_path = image_path
        self.image = cv2.imread(f"{image_path}")

        if resize:
            self.image = cv2.resize(self.image, (resize[0], resize[1]))

    def get_image_path(self):
        return self.image_path

    def _find_longest_list(self, lists: list[list]) -> tuple[list[float], int]:
        longest_list = max(lists, key=len, default=None)
        if longest_list:
            length_difference = len(longest_list) - len(min(lists, key=len))
            return longest_list, length_difference
        else:
            return None, 0

    def find_contour_segments(self) -> list[list[float, float, float, float]]:
        """
        This method finds the two largest contours in the image, approximates them to polygons,
        and extracts the line segments from these polygons.

        Parameters:
        None

        Returns:
        list[list[float]]: A list of two lists, where each inner list contains the line segments
                        of the corresponding contour. Each line segment is represented as a list
                        of four floats: [x1, y1, x2, y2], where (x1, y1) and (x2, y2) are the
                        coordinates of the start and end points of the line segment, respectively.
        """
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to create a binary track
        _, binary_track = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

        # Find contours in the binary track
        contours, _ = cv2.findContours(
            binary_track, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # Find the two largest contours
        largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        segments = [[], []]
        # Loop through the largest contours to extract the line segments
        for j, contour in enumerate(largest_contours):
            epsilon = 0.004 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Loop through the edges of the polygon to extract the line segments
            for i in range(len(approx)):
                p1 = approx[i][0]
                p2 = approx[(i + 1) % len(approx)][0]
                segments[j].append([p1[0], p1[1], p2[0], p2[1]])

        longest_segment, length_difference = self._find_longest_list(segments)

        if longest_segment:
            longest_segment[-1 - length_difference][-2:] = longest_segment[-1][-2:]
            del longest_segment[-length_difference:]

        return segments

    def find_segment_points(
        self, segments: list[float, float, float, float]
    ) -> list[tuple[float, float]]:
        """
        This method extracts the start points of each line segment from a list of line segments.

        Parameters:
        segments (list[float, float, float, float]): A list of line segments, where each segment is represented as a list of four floats: [x1, y1, x2, y2].

        Returns:
        list[tuple[float, float]]: A list of tuples representing the start points of each line segment. Each tuple contains the x and y coordinates of the start point.

        """
        return [(x1, y1) for x1, y1, x2, y2 in segments]

    def assign_closest_points(
        self, points1: list[tuple[float, float]], points2: list[tuple[float, float]]
    ) -> list[tuple[tuple[float, float]]]:
        """
        This function finds the closest point in 'points2' for each point in 'points1' and returns a list of these closest points.
        The function removes the assigned closest point from 'points2' to avoid duplicates.

        Parameters:
        points1 (list[tuple[float, float]]): A list of tuples representing the coordinates of points.
        points2 (list[tuple[float, float]]): A list of tuples representing the coordinates of points.

        Returns:
        list[list[tuple[float, float]]]: A list of tuples representing the closest points in 'points2' for each point in 'points1'.

        Raises:
        ValueError: If 'points1' or 'points2' is empty.
        """
        r: list[list[tuple[float, float]]] = []
        for point1 in points1:
            closest_point2 = sorted(
                [point2 for point2 in points2],
                key=lambda x: (x[0] - point1[0]) ** 2 + (x[1] - point1[1]) ** 2,
            )[0]
            r.append(tuple([point1, closest_point2]))
            points2.remove(closest_point2)
        return r
